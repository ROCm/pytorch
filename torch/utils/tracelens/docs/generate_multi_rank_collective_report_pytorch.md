# Generate Multi-Rank Collective Report

This utility analyzes **PyTorch JSON profile traces** from **multiple ranks** and produces a comprehensive **NCCL communication report** (Excel workbook and/or CSVs). 

---

## 🚀 Quick Start

Run the script with **either** a directory containing trace files **or** a trace file **pattern** with a single `*` placeholder for rank id.

### Directory mode
```bash
python generate_multi_rank_collective_report_pytorch.py   --trace_dir /path/to/traces   --world_size 8
```

### Pattern mode (strict rank substitution)
```bash
python generate_multi_rank_collective_report_pytorch.py   --trace_pattern "/logs/job123/rank*/trace.json"   --world_size 8
```

> The pattern must contain exactly one `*`, which is replaced with `0, 1, ..., world_size-1` to enumerate rank file paths.

### Installed entry point (if packaged)
If you install this script as part of a package with a console entry point, you can call it directly, e.g.:
```bash
tracelens-collectives   --trace_dir /path/to/traces   --world_size 8
```
_Replace `tracelens-collectives` with your actual entry point name if different._

---

## 📥 Input

- **PyTorch JSON traces** captured per-rank (e.g., via `torch.profiler` with JSON export).
- Naming patterns:
  - Directory mode (files inside `--trace_dir`): `rank0_trace.json`, `rank1_trace.json`, ...
  - Pattern mode: `--trace_pattern "/path/to/rank*_trace.json"`

---

## ⚙️ Command-Line Options

> Provide **either** `--trace_dir` **or** `--trace_pattern`.

| Argument | Default | Description |
|---|---|---|
| `--trace_dir` | `None` | Directory containing trace files (expects names like `rank*_trace.json`). |
| `--trace_pattern` | `None` | Template path with a **single** `*` placeholder for the rank id (strict substitution: `* → 0..world_size-1`). |
| `--world_size` | `Required` | Number of ranks in the distributed run.  |
| `--output_xlsx_path` | _auto-inferred_ | Path to save the Excel workbook. If omitted, a sensible default is created (see Output Behavior). |
| `--output_csvs_dir` | `None` | If provided, writes each sheet as an individual `.csv` under this directory. |
| `--detailed_analysis` | `False` | Include detailed per-rank sheets (see below). |
| `--agg_metrics` | `mean median min max` | Aggregations to compute in summaries. Allowed: `mean`, `median`, `min`, `max` (space-separated). |


---

## 📊 Excel Workbook Sheets

| Sheet Name | What it contains |
|---|---|
| `nccl_summary_implicit_sync` | Aggregated view of collectives that incurred implicit sync (AllReduce, AllGather, ReduceScatter, AllToAll balanced). Provides latency, skew, and aggregated bandwidth metrics (algorithm and bus). |
| `nccl_summary_long` | Aggregated NCCL operation stats per (rank, process group, collective, dtype, size). Includes counts and duration aggregates (sum/mean/std/min/max). |
| `nccl_long` | Per-event, per-rank NCCL records with timestamps, durations, stream, message sizes, and other attributes. Useful for drilling into specific slow ops. |
| `nccl_implicit_sync` | Per-collective implicit synchronization breakdown. Includes per-rank wait_time columns indicating time lost to implicit sync, plus timing/skew and bandwidth metrics. |
| `nccl_all2allv` | Detailed All2AllV analysis: variable send/recv sizes and splits per rank, with timing and skew columns per rank. |

_Notes:_
- Summary sheets (`nccl_summary_*`) are **always** produced; detailed per-event sheets (`nccl_long`, `nccl_implicit_sync`, `nccl_all2allv`) appear when `--detailed_analysis` is set.
- Column sets can evolve; the above reflects the provided example workbook.
---

## 📤 Output Behavior

- If **neither** `--output_xlsx_path` nor `--output_csvs_dir` is specified, the tool writes an Excel file to:
  - the provided `--trace_dir`, or
  - the **lowest common directory** of the expanded `--trace_pattern` file paths.
- If `--output_csvs_dir` is set, all sheets are written as individual CSV files under that directory.
- If `--output_xlsx_path` is set, a single Excel workbook containing all sheets is created.
- `openpyxl` is required for Excel; the tool will attempt to install it when missing.

**Lowest Common Directory (pattern mode)**  
When using `--trace_pattern`, the tool expands the pattern by replacing `*` with `0..world_size-1`, then computes the **lowest common ancestor directory** of those paths to pick a default output location.

---

