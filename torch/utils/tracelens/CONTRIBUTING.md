# Contributing to TraceLens

Thanks for your interest in improving **TraceLens** — a toolkit that parses PyTorch/JAX profiler traces and generates useful insights.

---

## 📋 Before You Start

- Read the [README](./README.md) to understand scope and architecture.
- Search existing **issues** and **discussions** to avoid duplicates.
- For new features and enhancements (new analyser, backend integration, refactor), **open an issue** first to align on approach.
- Prefer small, modular, focused PRs.
- **Have a ready-made utility?** If your utility is already developed, you can raise a PR to add it directly to `examples/custom_workflows/`. This lets the community start using it right away while we plan a tighter integration into the core library.
---

## ⚙️ Dev Setup

```bash
# clone
git clone https://github.com/AMD-AGI/TraceLens.git

# optional: virtual env
python3 -m venv .venv
source .venv/bin/activate

# install (editable) + dev extras
pip install -U pip
pip install -e .[dev]
```

## Project Structure (high level)
```
TraceLens/
├── TraceLens/
│   ├── Reporting/        # CLI tools for quick start utils
│   ├── Trace2Tree/       # Trace2Tree parses trace into tree data structure
│   ├── PerfModel/        # Op meta data parsing and performance modelling code (roofline, FLOPs/Byte, etc.)
│   ├── TreePerf/         # TreePerf uses Trace Tree and PerfModel to generate perf breakdowns and perf metrics TFLOPS/s, etc. 
|   |                     # This directory also contains GPUEventAnalyzer
│   ├── NcclAnalyser/     # Analysis of collective communications
│   ├── TraceFusion/      # Merging of multi‑rank traces into a global view
│   ├── TraceDiff/        # TraceDiff uses the Trace Tree format and does morphological comparison across traces
│   └── EventReplay/      # Extracts meta data and replays almost arbitrary operations
├── docs/               # tool-specific guides
├── examples/           # example traces, notebooks, scripts, custom-workflows
├── tests/              # unit & integration tests
└── setup.py
```
