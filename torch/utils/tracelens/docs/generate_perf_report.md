# Generate Performance Report

This Python script (`TraceLens/Reporting/generate_perf_report_pytorch.py`) processes a PyTorch JSON profile trace and outputs an Excel workbook or CSVs with relevant information.

---

## 🚀 Quick Start

Run the script with a profile JSON to generate an Excel report:

```bash
python generate_perf_report.py --profile_json_path path/to/profile.json 
```

Alternatively you can directly call the entry point with the same command line args,
 without the need to copy the script out of the repo. 

```bash
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/profile.json 
```

---

## 📋 Excel Workbook Sheets

| Sheet Name                  | Description                                                                                                      |
|----------------------------|------------------------------------------------------------------------------------------------------------------|
| `gpu_timeline`             | End-to-end GPU activity summary, including compute, memory copies, communication, and idle time.                 |
| `ops_summary_by_category`  | Summary of compute time grouped by operation category (e.g., GEMM, SDPA_fwd, elementwise).                       |
| `ops_summary`              | Summary of compute time at the individual operation level; each row corresponds to a unique operation name.      |
| `ops_all`                  | Detailed operation-level summary; each row corresponds to a unique (operation name, argument) combination.       |
| `short_kernels_histogram` | Histogram showing the distribution of kernel durations below the short-duration threshold.                       |
| `short_kernels_all_details`| Detailed list of short-duration kernels, including count, total/mean time, runtime percentage, and parent op.    |
| Roofline Sheets            | Roofline analysis for each operation category, including TFLOPs, TB/s, and FLOPs/byte metrics.                   |

---

## ⚙️ Optional Arguments

The script supports several optional arguments to customize the output report. By default, it generates an Excel file (`.xlsx`). If `--output_csvs_dir` is specified, individual CSV files are written instead.

| Argument                          | Default           | Description                                                                 |
|-----------------------------------|-------------------|-----------------------------------------------------------------------------|
| `--topk_ops N`                    | `None`            | Limit the number of rows in the unique-args launcher table.                |
| `--topk_short_kernels N`          | `None`            | Limit the number of rows in the short-kernel table.                         |
| `--topk_roofline_ops N`           | `None`            | Limit the number of rows in the roofline sheet.                             |
| `--extension_file`           | `None`            | Path to extension python file   
| `--short_kernel_study`            | `False`           | Include short-kernel analysis in the report.                                |
| `--short_kernel_threshold_us X`   | `10`              | Threshold (in microseconds) to classify a kernel as "short".               |
| `--short_kernel_histogram_bins B` | `100`             | Number of bins to use for the short-kernel duration histogram.             |
| `--output_xlsx_path PATH`         | `<auto-inferred>` | Path to save the Excel report. Auto-inferred if not provided.              |
| `--output_csvs_dir DIR`           | `None`            | If set, saves each sheet as a CSV file in the specified directory.         |

### 📦 Output Behavior

- If `--output_csvs_dir` is set, all output sheets are saved as individual CSV files in that directory.
- Otherwise, the script saves a single Excel file:
  - If `--output_xlsx_path` is not provided, it is inferred from the input JSON trace name (e.g., `profile.json` → `profile_perf_report.xlsx`).
- The `openpyxl` package is required only for the case when we write Excel files; it will be auto-installed if missing.

#### 🧪 Example Usage to write CSVs


```bash
python generate_perf_report.py \
  --profile_json_path traces/profile.json \
  --output_csvs_dir output_csvs/ \
  --topk_ops 50 \
```

## 🧩 Extensions: Custom Hooks for Tree and PerfModel

The `--extension_file` argument allows users to inject custom logic into the performance report generation pipeline. This is useful for experimenting with:

- Tree post-processing (e.g., injecting pseudo ops)
- Custom performance models for new op types
- Additional operation category definitions

### 🔧 How to Use

Pass a Python file path via `--extension_file`. The file can define one or more of the following optional symbols:

| Symbol Name                  | Type      | Description                                                                 |
|-----------------------------|-----------|-----------------------------------------------------------------------------|
| `tree_postprocess_extension`| `Callable`| Called with `perf_analyzer.tree`. Use to modify the tree structure post-construction. |
| `perf_model_extension`      | `dict`    | A mapping from op name (str) to a custom performance model class. These will override or extend existing models. |
| `dict_cat2names_extension`  | `dict`    | Mapping from new category names to lists of op names, merged into the built-in op categories. |

#### 📄 Example Extension File for MegatronLM in the examples dir

### ✅ Example Usage

```bash
python generate_perf_report.py \
  --profile_json_path traces/profile.json \
  --extension_file my_extension.py
```