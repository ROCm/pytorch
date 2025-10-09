# TraceLens
TraceLens is a Python library focused on **automating analysis from trace files** and enabling rich performance insights. Designed with **simplicity and extensibility** in mind, this library provides tools to simplify the process of profiling and debugging complex distributed training and inference systems.



## Key Features

✨ **Hierarchical Performance Breakdowns**: Pinpoint bottlenecks with a top-down view, moving from the overall GPU timeline (idle/busy) to operator categories (e.g., convolutions), individual operators, and right down to unique argument shapes.

⚙️ **Compute & Roofline Modeling**: Automatically translate raw timings into efficiency metrics like **TFLOP/s** and **TB/s** for popular operations. Determine if an op is compute- or memory-bound and see how effectively your code is using the hardware.

🔗 **Multi-GPU Communication Analysis**: Accurately diagnose scaling issues by dissecting collective operations. TraceLens separates pure communication time from synchronization skew and calculates effective bandwidth on your real workload, not a synthetic benchmark.

🔄 **Trace Comparison**: Quantify the impact of your changes with powerful trace diffing. By analyzing performance at the CPU dispatch level, TraceLens enables meaningful side-by-side comparisons across different hardware and software versions.

▶️ **Event Replay**: Isolate any operation for focused debugging. TraceLens generates minimal, self-contained replay scripts directly from trace metadata, making it simple to share IP-safe test cases with kernel developers.

🔧 **Extensible SDK**: Get started instantly with ready-to-use scripts, then build your own custom workflows using a flexible and hackable Python API.

## Quick Start
### Installation 

**1. Install TraceLens directly from GitHub:**
```bash 
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

**2. Command Line Scripts for popular analyses**
- **Generate Excel Reports from Traces** Detailed docs [here](docs/generate_perf_report.md) 
(you can use compressed traces too such as .zip and .gz)
```bash 
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/your/trace.json
```
- **Compare Traces** Detailed docs [here](docs/compare_perf_reports_pytorch.md)
```bash
TraceLens_compare_perf_reports_pytorch \
    baseline.xlsx \
    candidate.xlsx \
    --names baseline candidate \
    --sheets all \
    -o comparison.xlsx
```
- **Generate Collective Performance Report** Detailed docs [here](docs/generate_multi_rank_collective_report_pytorch.md)

```bash
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_dir /path/to/traces \
    --world_size 8 \
```


Refer to the individual module docs in the docs/ directory and the example notebooks under examples/ for further guidance.
**📦 Custom Workflows**: Check out [examples/custom_workflows/](examples/custom_workflows/) for community-contributed utilities including **roofline_analyzer** and **traceMap** — powerful tools we're working on integrating more tightly into the core library.

## Contributing

We welcome issues, bug reports, and pull requests. Feel free to open discussions in the GitHub repository
 or contribute new performance models, operator mappings or analysis modules. Please see CONTRIBUTING.md for guidelines.