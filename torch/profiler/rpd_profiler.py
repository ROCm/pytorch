# mypy: allow-untyped-defs
import json
import multiprocessing
import os
import sqlite3
import time
from collections.abc import Callable
from functools import partial
from typing import Any, Optional
from warnings import warn


def _monotonic_ns() -> int:
    """Return CLOCK_MONOTONIC in nanoseconds, matching rpdTracer's timestamps."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

from torch._C._autograd import DeviceType
from torch._C._profiler import (
    _rlog_set_record_shapes,
    _rlog_set_record_stacks,
    _rpd_available,
    _rpd_prepare_trace,
    _rpd_start_trace,
    _rpd_stop_trace,
    _rpd_trace_file_path,
)
from torch.autograd.profiler_util import (
    EventList,
    FunctionEvent,
)

from .profiler import (
    _default_schedule_fn,
    ProfilerAction,
)


__all__ = ["rpd_profile", "keep_trace"]

_trace_kept = False
_atexit_registered = False


def keep_trace() -> None:
    """Prevent the trace file from being deleted on process exit."""
    global _trace_kept
    _trace_kept = True


def _register_cleanup() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    _atexit_registered = True

    import atexit

    def _cleanup():
        if _trace_kept:
            return
        path = os.environ.get("RPDT_FILENAME", "")
        if path and os.path.exists(path):
            os.remove(path)

    atexit.register(_cleanup)



_CATEGORY_TO_SCOPE = {
    "function": 0,
    "backward_function": 1,
    "torchscript_function": 2,
    "kernel_function_dtype": 3,
    "custom_class": 4,
    "build_feature": 5,
    "lite_interpreter": 6,
    "user_scope": 7,
    "static_runtime_op": 8,
    "static_runtime_model": 9,
}


def _compute_flops(op_name, input_shapes):
    """Estimate FLOPs from op name and input shapes.

    Supports mm, addmm, bmm, baddbmm, mul, add.
    """
    if not input_shapes:
        return None

    if op_name in ("aten::mm", "aten::addmm"):
        # mm(mat1, mat2): sizes=[[M,K],[K,N]]
        # addmm(bias, mat1, mat2): sizes=[[N],[M,K],[K,N]]
        mat1 = input_shapes[-2] if len(input_shapes) >= 2 else None
        mat2 = input_shapes[-1] if len(input_shapes) >= 1 else None
        if not mat1 or not mat2 or len(mat1) != 2 or len(mat2) != 2:
            return None
        M, K = mat1
        _, N = mat2
        return 2 * M * K * N

    if op_name in ("aten::bmm", "aten::baddbmm"):
        mat1 = input_shapes[-2] if len(input_shapes) >= 2 else None
        mat2 = input_shapes[-1] if len(input_shapes) >= 1 else None
        if not mat1 or not mat2 or len(mat1) != 3 or len(mat2) != 3:
            return None
        B, M, K = mat1
        _, _, N = mat2
        return 2 * B * M * K * N

    if op_name in ("aten::mul", "aten::mul.Tensor", "aten::add", "aten::add.Tensor"):
        mat = input_shapes[0] if input_shapes else None
        if not mat:
            return None
        flops = 1
        for dim in mat:
            flops *= dim
        return flops

    return None


def _parse_args_json(args_str):
    """Parse the JSON args string written by rlog_client."""
    try:
        d = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return -1, None, None, []
    seq = d.get("seq", -1)
    op_id = d.get("op_id", None)
    sizes = d.get("sizes", None)
    stack = d.get("stack", [])
    return seq, op_id, sizes, stack


def _read_cpu_events(db_path, pid, start_ns, end_ns, use_device):
    """Read torch-domain API events from trace.rpd and return FunctionEvents."""
    events = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return EventList(events, use_device=use_device)

    try:
        cursor = conn.execute(
            """
            SELECT a.id, a.tid, a.start, a.end,
                   s.string AS apiName, c.string AS category, u.string AS args
            FROM rocpd_api a
            JOIN rocpd_string s ON s.id = a.apiName_id
            JOIN rocpd_string c ON c.id = a.category_id
            JOIN rocpd_ustring u ON u.id = a.args_id
            WHERE a.pid = ? AND a.start >= ? AND a.end <= ?
              AND a.domain_id IN (SELECT id FROM rocpd_string WHERE string = 'torch')
            ORDER BY a.start
            """,
            (pid, start_ns, end_ns),
        )

        for row in cursor:
            row_id, tid, start, end, api_name, category, args = row
            seq, op_id, sizes, stack = _parse_args_json(args)
            scope = _CATEGORY_TO_SCOPE.get(category, 0)

            fe = FunctionEvent(
                id=row_id,
                name=api_name,
                trace_name=api_name,
                thread=tid,
                fwd_thread=tid,
                start_us=start / 1000.0,
                end_us=end / 1000.0,
                sequence_nr=seq,
                input_shapes=sizes,
                stack=stack if stack else [],
                scope=scope,
                use_device=use_device,
                device_type=DeviceType.CPU,
                device_index=0,
                flops=_compute_flops(api_name, sizes),
            )
            events.append(fe)
    finally:
        conn.close()

    return events


def _attach_gpu_events(events, db_path, pid, start_ns, end_ns):
    """Read GPU ops from trace.rpd and attach as kernels on CPU FunctionEvents.

    Also creates separate FunctionEvent objects with device_type=CUDA for each
    GPU kernel, so that device time totals are computed correctly by _build_table.
    """
    if not events:
        return

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return

    try:
        cursor = conn.execute(
            """
            SELECT o.id, o.gpuId, o.queueId, o.start, o.end,
                   d.string AS description,
                   a.tid AS hip_tid, a.start AS hip_start, a.end AS hip_end
            FROM rocpd_op o
            JOIN rocpd_api_ops ao ON ao.op_id = o.id
            JOIN rocpd_api a ON a.id = ao.api_id
            JOIN rocpd_string d ON d.id = o.description_id
            WHERE a.pid = ? AND a.start >= ? AND a.start <= ?
            ORDER BY a.start
            """,
            (pid, start_ns, end_ns),
        )

        from collections import defaultdict

        by_thread: dict[int, list[FunctionEvent]] = defaultdict(list)
        for fe in events:
            by_thread[fe.thread].append(fe)

        for gpu_row in cursor:
            op_id, gpu_id, queue_id, gpu_start, gpu_end, description, hip_tid, hip_start, hip_end = (
                gpu_row
            )
            gpu_dur_us = (gpu_end - gpu_start) / 1000.0

            # Attach kernel to the innermost enclosing CPU ATen op
            thread_events = by_thread.get(hip_tid, [])
            best = None
            best_dur = float("inf")
            for fe in thread_events:
                fe_start_ns = fe.time_range.start * 1000.0
                fe_end_ns = fe.time_range.end * 1000.0
                if fe_start_ns <= hip_start and fe_end_ns >= hip_end:
                    dur = fe_end_ns - fe_start_ns
                    if dur < best_dur:
                        best = fe
                        best_dur = dur
            if best is not None:
                best.append_kernel(description, gpu_id, gpu_dur_us)
                best.is_legacy = True

            # Create a device-typed event for correct device time accounting
            gpu_fe = FunctionEvent(
                id=op_id,
                name=description,
                trace_name=description,
                thread=hip_tid,
                start_us=gpu_start / 1000.0,
                end_us=gpu_end / 1000.0,
                stack=[],
                use_device="cuda",
                device_type=DeviceType.CUDA,
                device_index=gpu_id,
                device_resource_id=queue_id,
            )
            events.append(gpu_fe)
    finally:
        conn.close()


class _RpdProfile:
    """Low-level profiler using rpdTracer for event collection.

    Analogous to _KinetoProfile but uses librpd_tracer.so instead of Kineto.
    CPU events are collected via rlog RecordFunction callbacks.
    GPU events are collected via roctracer/rocprofiler DataSources.
    All events are written to a trace.rpd sqlite database.
    """

    def __init__(
        self,
        *,
        activities=None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
    ) -> None:
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.use_device: Optional[str] = "cuda"

        self._start_ns: int = 0
        self._end_ns: int = 0
        self._pid: int = os.getpid()
        self._function_events: Optional[EventList] = None

    def start(self) -> None:
        self.prepare_trace()
        self.start_trace()

    def stop(self) -> None:
        self.stop_trace()

    def prepare_trace(self) -> None:
        _rpd_prepare_trace()

    def start_trace(self) -> None:
        self._function_events = None
        _register_cleanup()
        _rlog_set_record_shapes(self.record_shapes or self.with_flops)
        _rlog_set_record_stacks(self.with_stack)
        self._start_ns = _monotonic_ns()
        _rpd_start_trace()
        self._write_trace_metadata()

    def stop_trace(self) -> None:
        # Workaround for roctracer backend: GPU events are only delivered
        # after a device sync. Without this, GPU ops will be missing from
        # the trace. May be removable with a different backend.
        if self.use_device == "cuda":
            import torch.cuda
            torch.cuda.synchronize()
        self._end_ns = _monotonic_ns()
        _rpd_stop_trace()
        _rlog_set_record_shapes(False)
        _rlog_set_record_stacks(False)

    def events(self) -> EventList:
        if self._function_events is not None:
            return self._function_events

        if not _rpd_available():
            self._function_events = EventList([], use_device=self.use_device)
            return self._function_events

        db_path = _rpd_trace_file_path()
        cpu_events = _read_cpu_events(
            db_path, self._pid, self._start_ns, self._end_ns, self.use_device
        )
        _attach_gpu_events(cpu_events, db_path, self._pid, self._start_ns, self._end_ns)

        self._function_events = EventList(
            cpu_events,
            use_device=self.use_device,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        self._function_events._build_tree()
        return self._function_events

    def key_averages(
        self,
        group_by_input_shape: bool = False,
        group_by_stack_n: int = 0,
    ):
        return self.events().key_averages(group_by_input_shape, group_by_stack_n)

    def export_chrome_trace(self, path: str):
        """Export trace in Kineto-compatible Chrome JSON format."""
        from rocpd.tracing import generate_kineto_json

        conn = sqlite3.connect(self.trace_file_path())
        with open(path, "w") as f:
            generate_kineto_json(conn, f)
        conn.close()

    def export_rpd_chrome_trace(self, path: str):
        """Export trace in Chrome JSON format using rpd's native formatter.

        Produces richer output than export_chrome_trace, including GPU op
        tracks, API-to-op flow arrows, queue depth counters, and SMI data.
        """
        from rocpd.tracing import generate_rpd_json

        conn = sqlite3.connect(self.trace_file_path())
        with open(path, "w") as f:
            generate_rpd_json(conn, f)
        conn.close()

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """Export stack traces to a file for flamegraph visualization.

        Requires with_stack=True when creating the profiler.
        """
        return self.events().export_stacks(path, metric)

    def add_metadata(self, key: str, value: str) -> None:
        """Add a key-value metadata entry to the trace file."""
        db_path = _rpd_trace_file_path()
        try:
            conn = sqlite3.connect(db_path)
            conn.execute(
                "INSERT INTO rocpd_metadata(tag, value) VALUES (?, ?)",
                (f"torch.{key}", value),
            )
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            pass

    def add_metadata_json(self, key: str, value: str) -> None:
        """Add a key-value metadata entry with a JSON value to the trace file."""
        self.add_metadata(key, value)

    def _write_trace_metadata(self) -> None:
        if self.profile_memory:
            self.add_metadata("profile_memory", "1")
        if self.with_stack:
            self.add_metadata("with_stack", "1")
        if self.record_shapes:
            self.add_metadata("record_shapes", "1")
        if self.with_modules:
            self.add_metadata("with_modules", "1")
        if self.with_flops:
            self.add_metadata("with_flops", "1")

        dist_info = self._get_distributed_info()
        if dist_info:
            self.add_metadata("distributedInfo", json.dumps(dist_info))

    def _get_distributed_info(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        return {
            "pid": self._pid,
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
        }

    def trace_file_path(self) -> str:
        return _rpd_trace_file_path()


class rpd_profile(_RpdProfile):
    """Profiler context manager using rpdTracer.

    Usage::

        with torch.profiler.rpd_profile() as p:
            model(input)
        print(p.key_averages().table(sort_by="self_cpu_time_total"))

    With scheduling::

        with torch.profiler.rpd_profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=lambda p: print(p.key_averages().table()),
        ) as p:
            for step in range(N):
                train_step()
                p.step()
    """

    def __init__(
        self,
        *,
        activities=None,
        schedule: Callable[[int], ProfilerAction] | None = None,
        on_trace_ready: Callable[..., Any] | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
    ) -> None:
        super().__init__(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        )

        if schedule:
            self.schedule = schedule
        else:
            self.schedule = _default_schedule_fn
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)

        self.action_map: dict[
            tuple[ProfilerAction, ProfilerAction | None], list[Any]
        ] = {
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: WARMUP followed by NONE"),
                self.start_trace,
                self.stop_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [
                self.start_trace
            ],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: RECORD followed by NONE"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(warn, "Incorrect schedule: RECORD followed by WARMUP"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            # used for exit action
            (ProfilerAction.WARMUP, None): [self.start_trace, self.stop_trace],
            (ProfilerAction.RECORD, None): [self.stop_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                self.stop_trace,
                self._trace_ready,
            ],
        }

    def __enter__(self):
        self._transit_action(ProfilerAction.NONE, self.current_action)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._transit_action(self.current_action, None)

    def step(self) -> None:
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        self._transit_action(prev_action, self.current_action)

    def _trace_ready(self) -> None:
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action) -> None:
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()
