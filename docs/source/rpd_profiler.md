# RPD Profiler for PyTorch

`torch.profiler.rpd_profile` is an alternative profiler that uses
[rpdTracer](https://github.com/ROCm/rocmProfileData) instead of Kineto for
event collection. It provides the same Python API as `torch.profiler.profile`
but with a simpler architecture and lower overhead. rpdTracer supports both
ROCm (via roctracer/rocprofiler) and CUDA (via CUPTI).

## Quick Start

```python
import torch
from torch.profiler import rpd_profile

with rpd_profile() as p:
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

print(p.key_averages().table(sort_by="self_cpu_time_total"))
```

## How It Works

The standard `torch.profiler.profile` uses Kineto (libkineto) and builds a
complex in-memory event pipeline: RecordFunction callbacks collect ATen ops
into per-thread queues, Kineto collects GPU events via CUPTI/roctracer, and
post-processing merges them into a unified trace.

`rpd_profile` replaces this entire pipeline. CPU ATen ops are already captured
by rlog (a lightweight RecordFunction callback registered at module load).
GPU kernel and memory copy events are captured by rpdTracer's internal
DataSources (roctracer/rocprofiler on ROCm, CUPTI on CUDA). All events are written to a shared
SQLite database (`trace.rpd`). The profiler simply controls when collection
is active and reads results from the database.

```
torch.profiler.profile (Kineto)          torch.profiler.rpd_profile
================================         ================================
RecordFunction callbacks                 rlog_client (already registered)
  -> per-thread RecordQueue                -> rlog -> RlogDataSource
Kineto GPU collection (CUPTI)            rpdTracer GPU collection (roctracer/CUPTI)
  -> in-memory ActivityTrace               -> sqlite (trace.rpd)
C++ post-processing & merging            SQL queries at reporting time
Python FunctionEvent creation            Python FunctionEvent creation
```

## Features

### Supported

| Feature | `profile()` | `rpd_profile()` | Notes |
|---------|-------------|-----------------|-------|
| CPU operator timing | Yes | Yes | Via rlog RecordFunction callbacks |
| GPU kernel timing | Yes | Yes | Via roctracer (ROCm) or CUPTI (CUDA) DataSources |
| CPU-GPU correlation | Yes | Yes | Time-nesting match through HIP layer |
| `key_averages().table()` | Yes | Yes | Same EventList/FunctionEvent classes |
| `export_chrome_trace()` | Yes | Yes | Uses EventList Python formatter |
| `export_stacks()` | Yes | Yes | Requires `with_stack=True` |
| `record_shapes` | Yes | Yes | Dynamically toggled per session |
| `with_stack` | Yes | Yes | Captures Python frames per op |
| `with_flops` | Yes | Partial | Supports mm, addmm, bmm, mul, add (not conv2d) |
| Schedule (wait/warmup/active) | Yes | Yes | Same state machine |
| `on_trace_ready` callback | Yes | Yes | Same pattern |
| `add_metadata()` | Yes | Yes | Written to `rocpd_metadata` table with `torch.` prefix |
| Multi-process tracing | Yes | Yes | All processes write to same `.rpd` file |
| `export_rpd_chrome_trace()` | No | Yes | Rich output via rocpd (GPU tracks, flow arrows, queue depth) |

### Not Supported

| Feature | Reason |
|---------|--------|
| `toggle_collection_dynamic()` | rpd uses start/stop, not per-activity toggling |
| `export_memory_timeline()` | Deprecated in `profile()`, not implemented |
| `with_modules` | rlog does not capture module hierarchy |
| `profile_memory` | rlog does not capture memory events |
| conv2d FLOPs | Requires extra args (padding, stride, dilation) not captured by rlog |

## Trace File Management

When `RPDT_FILENAME` is not set, `import torch` generates a default trace
file named `torch_trace_{pid}.rpd`, deletes any existing file with that name,
and sets `RPDT_FILENAME` in the environment. The file is cleaned up on
process exit unless `keep_trace()` is called.

When `RPDT_FILENAME` is already set (e.g. by a parent process, torchrun,
or the user), the file is left as-is and new events are appended. Child
processes inherit `RPDT_FILENAME` from their parent and will not delete or
recreate the file.

```python
from torch.profiler.rpd_profiler import keep_trace

# Prevent automatic cleanup — file persists after exit
keep_trace()
```

### Custom filename

Set `RPDT_FILENAME` before import:

```bash
RPDT_FILENAME=my_trace.rpd python train.py
```

### With torchrun

```bash
torchrun --profile-url trace.rpd --nproc-per-node 8 train.py
```

This deletes the old file, sets `RPDT_FILENAME` and `PROFILE_URL` in the
environment, and all workers write to the same file. Events are disambiguated
by PID; multi-node traces use `pid_stride` and `gpu_stride` metadata.

## Overhead

**During profiling:** Zero added overhead to the per-op hot path. The rlog
RecordFunction callbacks already exist regardless of whether `rpd_profile` is
used. The profiler only controls `rpdstart()`/`rpdstop()` at session
boundaries.

**After profiling:** Event reading and correlation happens via SQL queries
against the `.rpd` file. This is post-processing that does not affect the
profiled workload.

## Architecture

```
torch/profiler/rpd_profiler.py      Python API (_RpdProfile, rpd_profile)
torch/csrc/profiler/rpd_shim.h/cpp  C++ shim: dlopen librpd_tracer.so
torch/csrc/profiler/python/init.cpp pybind bindings
torch/csrc/profiler/standalone/     rlog_client.cpp (RecordFunction callbacks)
                                    rlog_client.h (record_shapes/stacks setters)
```

The shim uses `dlopen` to load `librpd_tracer.so` at runtime. If the library
is not available, all profiler operations are safe no-ops. The shim resolves
three function pointers: `rpdstart`, `rpdstop`, `rpdflush`.

## Reading the trace.rpd file directly

The `.rpd` file is a standard SQLite database. You can query it directly:

```bash
sqlite3 torch_trace_1234.rpd "SELECT * FROM api LIMIT 10"
sqlite3 torch_trace_1234.rpd "SELECT * FROM top"
sqlite3 torch_trace_1234.rpd "SELECT * FROM kernel LIMIT 10"
```

See the [rocmProfileData documentation](https://github.com/ROCm/rocmProfileData)
for the full schema and available views.
