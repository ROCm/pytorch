import inspect
import json
import os
import statistics
import time
from functools import cached_property, wraps
from itertools import chain
from statistics import median
from typing import Any, Callable, Optional, Union
from typing_extensions import Concatenate, ParamSpec, Self, TypeVar

import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.config import (
    use_experimental_benchmarker,
    use_torch_profiler_benchmarker,
)


logger = torch._logging.getArtifactLogger(__name__, "benchmarking")
use_experimental_benchmarker = (
    use_experimental_benchmarker and torch.cuda.is_available()
)
use_torch_profiler_benchmarker = (
    use_torch_profiler_benchmarker and torch.cuda.is_available()
)


MILLISECONDS_PER_SECOND = 1000

P = ParamSpec("P")
T = TypeVar("T")


def time_and_count(
    fn: Callable[Concatenate[Any, P], T],
) -> Callable[Concatenate[Any, P], T]:
    """Wraps `fn` with `dynamo_timed` context, and increments the appropriate dynamo
    counters. It is expected that `fn` is a method of `Benchmarker` or one of its
    subclasses; typing limitations prevent us from declaring this directly.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
        fn_qual_name = f"{self.__class__.__name__}.{fn.__name__}"
        counters["inductor"][f"benchmarking.{fn_qual_name}"] += 1
        with dynamo_timed(fn_qual_name, log_pt2_compile_event=False):
            return fn(self, *args, **kwargs)

    return wrapper


class Benchmarker:
    def __init__(self: Self) -> None:
        pass

    @time_and_count
    def benchmark(
        self: Self,
        fn: Callable[..., Any],
        fn_args: tuple[Any, ...],
        fn_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> float:
        """Benchmark `fn(*fn_args, *fn_kwargs)` and return the runtime, in milliseconds (the
        actual runtime calculation is dictated by the benchmarking implementation, but may be
        one of [mean, median, minimum, etc.]). Functions as a convenience wrapper around
        device-specific implementations, like `benchmark_cpu` and `benchmark_gpu`. Raises
        `ValueError(...)` if we can't safely infer the device type of `fn`; for example,
        if multiple device types are found in `fn_args` and `fn_kwargs`, or if no device
        types are found.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - **kwargs: The benchmarking implementation's kwargs.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
        inferred_device = None
        for arg_or_kwarg in chain(fn_args, fn_kwargs.values()):
            if not isinstance(arg_or_kwarg, torch.Tensor):
                continue
            if inferred_device is None:
                inferred_device = arg_or_kwarg.device
            elif arg_or_kwarg.device != inferred_device:
                raise ValueError(
                    "Can't safely infer the device type of `fn` with multiple device types in `fn_args` and `fn_kwargs`!"
                )
        if inferred_device is None:
            raise ValueError(
                "Can't safely infer the device type of `fn` with no device types in `fn_args` or `fn_kwargs`! You should be calling `.benchmark_cpu` or `.benchmark_gpu` directly."  # noqa: B950
            )
        _callable = lambda: fn(*fn_args, **fn_kwargs)  # noqa: E731
        if inferred_device == torch.device("cpu"):
            return self.benchmark_cpu(_callable, **kwargs)
        # TODO(nmacchioni): For non-CPU functions we default to using the GPU-specific benchmarking
        # implementation which was written specifically with CUDA devices in mind, we may want to
        # explore alternate implementations for other device types.
        return self.benchmark_gpu(_callable, **kwargs)

    @time_and_count
    def benchmark_cpu(
        self: Self, _callable: Callable[[], Any], warmup: int = 20, rep: int = 100
    ) -> float:
        """Benchmark the CPU callable, `_callable`, and return the median runtime,
        in milliseconds.

        Arguments:
        - _callable: The CPU callable to benchmark.

        Keyword Arguments:
        - warmup: Optionally, the duration, in milliseconds, to run `_callable`
        before benchmarking starts.
        - rep: Optionally, the duration, in milliseconds, to run `_callable`
        during benchmarking.

        Returns:
        - The median runtime of `_callable`, in milliseconds.
        """

        def run_for(ms: int) -> list[float]:
            timings = []
            run_start_t = time.perf_counter()
            while True:
                start_t = time.perf_counter()
                _callable()
                end_t = time.perf_counter()
                timings.append((end_t - start_t) * MILLISECONDS_PER_SECOND)
                if ((end_t - run_start_t) * MILLISECONDS_PER_SECOND) > ms:
                    break
            return timings

        run_for(warmup)
        return median(run_for(rep))

    @time_and_count
    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


class TritonBenchmarker(Benchmarker):
    @cached_property
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
        try:
            from triton.testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @time_and_count
    def benchmark_gpu(self: Self, _callable: Callable[[], Any], **kwargs: Any) -> float:
        """Benchmark the GPU callable, `_callable`, and return the runtime, in milliseconds.

        Arguments:
        - _callable: The GPU callable to benchmark.

        Keyword Arguments:
        - quantiles: Optionally, a tuple of floats denoting the requested quantiles.
        - return_mode: Optionally, the requested return mode. Currently, Triton's
        `do_bench` supports min, max, mean, and median return modes.
        - **kwargs: Additional kwargs passed to Triton's `do_bench`.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        """
        do_bench_params = inspect.signature(self.triton_do_bench).parameters
        for kwarg in list(kwargs.keys()):
            if kwarg not in do_bench_params:
                del kwargs[kwarg]
        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        elif "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


class InductorBenchmarker(TritonBenchmarker):  # noqa: docstring_linter
    @cached_property
    def L2_cache_size(self: Self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return props.L2_cache_size

    def get_event_pairs(
        self: Self, iters: int
    ) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Get `iters` pairs of CUDA events."""
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]

    def get_event_pairs_min_timing(
        self: Self, event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ) -> float:
        """Get the minimum timing, in milliseconds, for a group of CUDA event pairs."""
        return min(
            [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
        )

    @time_and_count
    def benchmark_gpu(  # type: ignore[override]
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = 5,
        memory_warmup_iters: int = 100,
        benchmark_iters: int = 100,
        max_benchmark_duration: int = 25,
        return_mode: str = "min",
        grad_to_none: Optional[list[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Union[float, list[float]]:
        """Benchmark a GPU callable using a custom benchmarking implementation.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - estimation_iters: Optionally, the number of iterations to run `_callable`
        during runtime estimation.
        - memory_warmup_iters: Optionally, the number of iterations to flush the L2
        cache before starting benchmarking.
        - benchmark_iters: Optionally, the number of iterations to run `_callable`
        during the benchmarking.
        - max_benchmark_duration: Optionally, the maximum duration of the benchmarking,
        in milliseconds. An estimated duration is calculated based on the values
        of `memory_warmup_iters` and `benchmark_iters`, along with the estimated
        runtime of `_callable` and various other factors, and we then shrink
        `benchmark_iters` to fit in the allotted maximum duration.
        - return_mode: Return mode for benchmark results. Options are "min" (default),
        "all" (returns all measurements).
        - grad_to_none: Optionally, a list of tensors whose gradients should be cleared
        before each benchmark iteration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - If return_mode="min": The minimum runtime of `_callable`, in milliseconds.
        - If return_mode="all": List of all runtime measurements, in milliseconds.
        """
        # we don't want any outside errors propagating into benchmarking
        torch.cuda.synchronize()

        # warmup `_callable` (and catches any failures in the process)
        _callable()
        torch.cuda.synchronize()

        # see https://github.com/triton-lang/triton/pull/840 for why `dtype=torch.int`
        buffer = torch.empty(self.L2_cache_size // 4, dtype=torch.int, device="cuda")
        buffer.zero_()

        # estimate the runtime of `_callable`
        event_pairs = self.get_event_pairs(estimation_iters)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()
        estimated_timing = self.get_event_pairs_min_timing(event_pairs)

        # adjust `benchmark_iters` to fit in the maximum benchmarking duration
        benchmark_iters = max(
            min(benchmark_iters, int(max_benchmark_duration // estimated_timing)), 1
        )

        # do the memory warmup
        for _ in range(memory_warmup_iters):
            buffer.zero_()

        # benchmark `_callable`
        event_pairs = self.get_event_pairs(benchmark_iters)
        for start_event, end_event in event_pairs:
            # Clear gradients before timing (matches triton.testing.do_bench)
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()

        # explicitly delete the buffer, sometimes helps memory
        # footprint metrics in OSS Inductor performance benchmarks
        del buffer

        # Return based on the requested mode
        if return_mode == "all":
            # Get all timings from event pairs
            all_timings = [
                start_event.elapsed_time(end_event)
                for start_event, end_event in event_pairs
            ]
            return all_timings
        elif return_mode == "min":
            benchmarked_timing = self.get_event_pairs_min_timing(event_pairs)
            # return the minimum of `estimated_timing` and `benchmarked_timing`,
            # we just want the minimum timing overall so we might as well check both
            return min(estimated_timing, benchmarked_timing)
        else:
            raise ValueError(
                f"Unsupported return_mode: {return_mode}. Use 'min' or 'all'."
            )


_CALLABLE_PROFILE_EVENT_NAME = "_CALLABLE"


def _annotation_sample_us(times):
    """Drop a ROCTracer span that covers host idle, then return the mean in us.

    On this ROCm build the first ``_CALLABLE`` device annotation can last
    hundreds of milliseconds while the real kernels are microseconds. The
    upstream mean of every annotation is then dominated by that one span.
    Samples larger than 50x the median are left out. Returns None when no
    positive sample remains.
    """
    positive = [t for t in times if t > 0]
    if not positive:
        return None
    if len(positive) >= 2:
        ordered = sorted(positive)
        median = ordered[len(ordered) // 2]
        if median > 0:
            kept = [t for t in positive if t <= median * 50.0]
            if kept:
                positive = kept
    return sum(positive) / len(positive)


def _get_callable_device_kernel_time_us(kineto_events, profiler_events, rep):
    """Sum device kernels correlated with ``_CALLABLE``, in microseconds.

    Ported from the newer TorchProfilerBenchmarker fallback. This Kineto
    build has no ``activity_type()``; ``is_user_annotation()`` identifies the
    ``_CALLABLE`` span that must not be added as if it were a kernel.
    """
    from torch.autograd import DeviceType

    benchmark_event_ids = set()

    def collect_cpu_event_ids(event):
        if event.device_type != DeviceType.CPU:
            return
        benchmark_event_ids.add(event.id)
        for child in event.cpu_children:
            collect_cpu_event_ids(child)

    benchmark_events = [
        event
        for event in profiler_events
        if event.name == _CALLABLE_PROFILE_EVENT_NAME
        and event.device_type == DeviceType.CPU
    ]
    if len(benchmark_events) != rep:
        raise RuntimeError(
            "Expected %d %s profiling events. Found %d events."
            % (rep, _CALLABLE_PROFILE_EVENT_NAME, len(benchmark_events))
        )

    for event in benchmark_events:
        collect_cpu_event_ids(event)

    callable_kineto_windows = []
    for ev in kineto_events:
        if (
            ev.name() == _CALLABLE_PROFILE_EVENT_NAME
            and ev.device_type() == DeviceType.CPU
        ):
            callable_kineto_windows.append((ev.start_ns(), ev.end_ns()))

    if callable_kineto_windows:
        for ev in kineto_events:
            if ev.device_type() != DeviceType.CPU:
                continue
            ev_start = ev.start_ns()
            ev_end = ev.end_ns()
            for win_start, win_end in callable_kineto_windows:
                if ev_start >= win_start and ev_end <= win_end:
                    benchmark_event_ids.add(ev.correlation_id())
                    break

    device_time_us = 0.0
    for event in kineto_events:
        if event.device_type() != DeviceType.CUDA:
            continue
        if getattr(event, "is_user_annotation", lambda: False)():
            continue
        linked_correlation_id = event.linked_correlation_id()
        correlation_id = event.correlation_id()
        if event.name() == "Context Sync":
            continue
        if linked_correlation_id in benchmark_event_ids or (
            linked_correlation_id == 0 and correlation_id in benchmark_event_ids
        ):
            device_time_us += (event.end_ns() - event.start_ns()) / 1000.0
    return device_time_us


def _tpb_context(fn):
    """Kernel name and launcher config closed over by CachingAutotuner.bench."""
    kernel = ""
    config = ""
    try:
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            meta = getattr(obj, "inductor_meta", None)
            if isinstance(meta, dict) and meta.get("kernel_name"):
                kernel = meta["kernel_name"]
            cfg = getattr(obj, "config", None)
            if cfg is not None and hasattr(cfg, "kwargs"):
                kw = cfg.kwargs or {}
                config = (
                    "XBLOCK=%s,R0_BLOCK=%s,RBLOCK=%s,num_warps=%s,num_stages=%s"
                    % (
                        kw.get("XBLOCK"),
                        kw.get("R0_BLOCK"),
                        kw.get("RBLOCK"),
                        getattr(cfg, "num_warps", None),
                        getattr(cfg, "num_stages", None),
                    )
                )
    except Exception:
        logger.warning("TPB_ITER context failed", exc_info=True)
    return kernel, config


def _robust_center_us(samples):
    """Median after dropping the startup sample and points beyond 3 MAD.

    The first sample is the only one that is systematically high, and only
    on short kernels. The second sample already sits on the later median.
    The 8% floor stays wider than 3 * 1.4826 * MAD for kernels at or above
    about 0.01 ms, so normal jitter is kept. If the MAD gate keeps fewer
    than 70% of the post-startup samples, fall back to the pre-gate median.
    """
    xs = [float(s) for s in samples if s and s > 0]
    dropped = 0
    if len(xs) >= 3:
        xs = xs[1:]
        dropped += 1
    if not xs:
        return None, dropped, False
    med = statistics.median(xs)
    devs = sorted(abs(x - med) for x in xs)
    mad = devs[len(devs) // 2]
    robust_sigma = 1.4826 * mad
    limit = max(3.0 * robust_sigma, 0.08 * med) if med > 0 else 0.0
    kept = [x for x in xs if abs(x - med) <= limit]
    if not kept or len(kept) < 0.7 * len(xs):
        return med, dropped, True
    dropped += len(xs) - len(kept)
    return statistics.median(kept), dropped, False


def _per_iteration_device_kernel_us(kineto_events, profiler_events, kernel_name):
    """GPU-kernel microseconds for each ``_CALLABLE`` iteration.

    The device ``_CALLABLE`` annotation covers the whole host callable,
    including argument clones and memsets. Keep device kernels correlated
    with that iteration, preferring events whose name contains ``kernel_name``,
    and return the longest one. That matches rocprof AverageNs for one launch.
    """
    from torch.autograd import DeviceType

    benchmark_events = [
        event
        for event in profiler_events
        if event.name == _CALLABLE_PROFILE_EVENT_NAME
        and event.device_type == DeviceType.CPU
    ]

    def ids_for(event):
        found = set()

        def walk(node):
            if node.device_type != DeviceType.CPU:
                return
            found.add(node.id)
            for child in node.cpu_children:
                walk(child)

        walk(event)
        return found

    per_ids = [ids_for(event) for event in benchmark_events]
    windows = []
    for ev in kineto_events:
        if (
            ev.name() == _CALLABLE_PROFILE_EVENT_NAME
            and ev.device_type() == DeviceType.CPU
        ):
            windows.append((ev.start_ns(), ev.end_ns()))
    windows.sort()
    if windows and len(windows) == len(per_ids):
        for ev in kineto_events:
            if ev.device_type() != DeviceType.CPU:
                continue
            start_ns = ev.start_ns()
            end_ns = ev.end_ns()
            for index, (win_start, win_end) in enumerate(windows):
                if start_ns >= win_start and end_ns <= win_end:
                    per_ids[index].add(ev.correlation_id())
                    break

    def is_utility(name):
        low = name.lower()
        return any(
            token in low
            for token in ("memset", "memcpy", "copy_", "zero", "fill", "context sync")
        )

    buckets = [[] for _ in per_ids]
    for event in kineto_events:
        if event.device_type() != DeviceType.CUDA:
            continue
        if getattr(event, "is_user_annotation", lambda: False)():
            continue
        name = event.name()
        if name == _CALLABLE_PROFILE_EVENT_NAME or is_utility(name):
            continue
        duration_us = (event.end_ns() - event.start_ns()) / 1000.0
        linked = event.linked_correlation_id()
        correlation = event.correlation_id()
        for index, ids in enumerate(per_ids):
            if linked in ids or (linked == 0 and correlation in ids):
                buckets[index].append((name, duration_us))
                break

    samples = []
    for items in buckets:
        named = [dt for name, dt in items if kernel_name and kernel_name in name]
        chosen = named if named else [dt for _name, dt in items]
        # rocprof AverageNs is one launch, not the sum of clones and memsets
        # that share the host callable.
        samples.append(max(chosen) if chosen else 0.0)
    return samples


def _tpb_record(record):
    """Print one bench record and append it to TPB_ITER_LOG when set."""
    try:
        line = json.dumps(record, ensure_ascii=True)
        print("TPB_ITER " + line, flush=True)
        path = os.environ.get("TPB_ITER_LOG")
        if path:
            with open(path, "a") as fh:
                fh.write(line + "\n")
    except Exception:
        logger.warning("TPB_ITER log failed", exc_info=True)


class TorchProfilerBenchmarker(InductorBenchmarker):  # noqa: docstring_linter
    """Benchmarker that uses torch.profiler for GPU kernel benchmarking.

    Synced from the newer pytorch TorchProfilerBenchmarker. Device-agnostic
    helpers in that file (get_interface_for_device, activity_type,
    gpu_benchmark_lock) are not in this install. Timing stays on the CUDA
    profiler activity, which is how ROCm Kineto reports HIP kernels here.
    A single ROCTracer annotation that dwarfs the median is dropped before
    the mean; if no annotation remains, device kernels correlated with
    _CALLABLE are summed instead.
    """

    @time_and_count
    def benchmark_gpu(  # type: ignore[override]
        self: Self,
        _callable: Callable[[], Any],
        warmup: int = 25,
        rep: int = 100,
        estimation_iters: int = 5,
        memory_warmup_iters: int = 10,
        max_benchmark_duration: int = 25,
        return_mode: str = "mean",
        grad_to_none: Optional[list[torch.Tensor]] = None,
        device_type: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Benchmark a GPU callable using torch.profiler.

        Times the ``_CALLABLE`` device annotation. ``device_type`` is accepted
        for call compatibility and ignored; this build's profiler activity is CUDA.
        """
        del device_type
        # bench() passes rep=40. TORCHINDUCTOR_PROFILER_BENCH_REP replaces that
        # count. The 25 ms budget would otherwise clamp a 0.4 ms kernel near
        # 62 iterations, so a larger rep also lifts max_benchmark_duration.
        caller_rep = int(rep)
        warmup_param = int(warmup)
        rep_env = os.environ.get("TORCHINDUCTOR_PROFILER_BENCH_REP")
        if rep_env:
            rep = max(1, int(rep_env))
        if rep > 40:
            max_benchmark_duration = max(int(max_benchmark_duration), int(rep * 0.5))
        kernel_name, config_desc = _tpb_context(_callable)
        torch.cuda.synchronize()

        bare_start = torch.cuda.Event(enable_timing=True)
        bare_end = torch.cuda.Event(enable_timing=True)
        bare_start.record()
        _callable()
        bare_end.record()
        torch.cuda.synchronize()
        bare_ms = bare_start.elapsed_time(bare_end)

        # Keep Triton's 256 MB cache flush on ROCm. On other backends, reuse
        # the shared L2-sized flush from InductorBenchmarker.
        if torch.version.hip:
            buffer_size_bytes = 256 * 1024 * 1024
        else:
            buffer_size_bytes = self.L2_cache_size
        buffer = torch.empty(buffer_size_bytes // 4, dtype=torch.int, device="cuda")
        buffer.zero_()

        event_pairs = self.get_event_pairs(estimation_iters)
        for start_event, end_event in event_pairs:
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            buffer.zero_()
            start_event.record()
            _callable()
            end_event.record()
        torch.cuda.synchronize()
        estimate_ms_each = [
            start_event.elapsed_time(end_event) for start_event, end_event in event_pairs
        ]
        estimated_ms = min(estimate_ms_each) if estimate_ms_each else 0.0
        requested_rep = int(rep)
        if estimated_ms > 0:
            rep = max(min(rep, int(max_benchmark_duration / estimated_ms)), 1)

        # Flush L2 once, then warm the kernel's own working set. Live rocprof
        # sees this kernel with cache left by the previous op, not after a
        # 256 MB wipe on every launch. The scored loop does not flush.
        buffer.zero_()
        torch.cuda.synchronize()
        for _ in range(3):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            _callable()
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
        ) as prof:
            for _ in range(rep):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                with torch.profiler.record_function(_CALLABLE_PROFILE_EVENT_NAME):
                    _callable()

        torch.cuda.synchronize()

        from torch.autograd import DeviceType as _DeviceType

        annotation_us = []
        for kineto_event in prof.profiler.kineto_results.events():
            if (
                kineto_event.name() == _CALLABLE_PROFILE_EVENT_NAME
                and kineto_event.device_type() == _DeviceType.CUDA
            ):
                annotation_us.append(
                    (kineto_event.end_ns() - kineto_event.start_ns()) / 1000.0
                )

        kernel_us_each = _per_iteration_device_kernel_us(
            prof.profiler.kineto_results.events(),
            prof.events(),
            kernel_name,
        )
        kernel_center_us, kernel_dropped, kernel_pre_gate = _robust_center_us(
            kernel_us_each
        )
        annotation_center_us, annotation_dropped, annotation_pre_gate = (
            _robust_center_us(annotation_us)
        )
        if kernel_center_us is not None and kernel_center_us > 0:
            per_iter_us = kernel_center_us
            dropped_outlier_n = kernel_dropped
            score_source = (
                "kernel_pre_gate_median" if kernel_pre_gate else "kernel_robust_median"
            )
        elif annotation_center_us is not None and annotation_center_us > 0:
            per_iter_us = annotation_center_us
            dropped_outlier_n = annotation_dropped
            score_source = (
                "annotation_pre_gate_median"
                if annotation_pre_gate
                else "annotation_robust_median"
            )
        else:
            raise AssertionError(
                "TorchProfilerBenchmarker: no GPU kernel time or _CALLABLE "
                "annotation for %s" % (kernel_name or "unknown")
            )

        avg_time_ms = per_iter_us / 1000.0
        _tpb_record(
            {
                "kernel": kernel_name,
                "config": config_desc,
                "warmup_param": warmup_param,
                "warmup_param_used": False,
                "bare_launch_n": 1,
                "bare_launch_ms": [round(bare_ms, 4)],
                "memory_warmup_iters": int(memory_warmup_iters),
                "memory_warmup_timed": False,
                "l2_flush_each_bench_iter": False,
                "estimate_n": len(estimate_ms_each),
                "estimate_ms": [round(v, 4) for v in estimate_ms_each],
                "caller_rep": caller_rep,
                "requested_rep": requested_rep,
                "max_benchmark_duration_ms": int(max_benchmark_duration),
                "rep": int(rep),
                "annotation_n": len(annotation_us),
                "annotation_ms": [round(v / 1000.0, 4) for v in annotation_us],
                "kernel_ms": [round(v / 1000.0, 4) for v in kernel_us_each],
                "kernel_center_us": None if kernel_center_us is None else round(kernel_center_us, 4),
                "kernel_dropped": kernel_dropped,
                "annotation_center_us": None
                if annotation_center_us is None
                else round(annotation_center_us, 4),
                "annotation_dropped": annotation_dropped,
                "dropped_outlier_n": dropped_outlier_n,
                "bench_ms": [round(v / 1000.0, 4) for v in kernel_us_each],
                "pre_gate_fallback": score_source.endswith("pre_gate_median"),
                "score_ms": round(avg_time_ms, 4),
                "score_source": score_source,
            }
        )
        del buffer

        if return_mode in ("min", "mean", "max"):
            return avg_time_ms
        raise ValueError(
            "Unsupported return_mode: %s. Use 'min', 'mean', or 'max'." % return_mode
        )


benchmarker = (
    TorchProfilerBenchmarker()
    if use_torch_profiler_benchmarker
    else (
        InductorBenchmarker() if use_experimental_benchmarker else TritonBenchmarker()
    )
)
