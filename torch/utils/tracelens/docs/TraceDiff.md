# TraceDiff

TraceDiff is a Python API and a component of TraceLens for comparing two PyTorch Kineto trace files. It allows users to identify, visualize, and analyze differences between execution traces at both the operation and kernel levels.

Unlike a simple leaf-level operation comparison, TraceDiff considers the morphological structure of traces to automatically determine the lowest common node. This improves accuracy when dealing with differences in operator lowering at the host call stack — for example, `aten::convolution` lowering to `aten::miopen_convolution` on ROCm and `aten::cudnn_convolution` on CUDA. A leaf-level comparison alone would treat these as completely different operations, whereas TraceDiff can automatically match them at the appropriate level.

TraceDiff is particularly useful for regression analysis, performance debugging, and assessing the impact of code or environment changes on GPU workloads.


---

## Key Features

- **Automated Tree Comparison**: Builds hierarchical event trees from two traces and identifies points of difference (PODs) using a recursive diff algorithm.
- **Tree Diff Visualization**: Produces a diff output file that highlights matched and unmatched operations between traces.
- **Detailed and Summary Reports**: Generates CSV reports with kernel time statistics and aggregated summaries for each operation.
- **UID Mapping**: Provides a mapping between event UIDs in the two traces, enabling cross-referencing and deeper analysis.
- **Seamless Integration**: Designed to work with TraceLens's TraceToTree objects and PyTorch profiler JSON traces.

---

## Quick Start


### Example: Compare Two Traces and Generate Reports

```python
from TraceLens import TreePerfAnalyzer, TraceDiff
import json

# Load two trace files into tree perf analyzer
trace_file1 = "/path/to/trace1.json"
trace_file2 = "/path/to/trace2.json"

perf_analyzer1 = TreePerfAnalyzer.from_file(trace_file1)
perf_analyzer2 = TreePerfAnalyzer.from_file(trace_file2)
tree1 = perf_analyzer1.tree
tree2 = perf_analyzer2.tree

# Compare and analyze the trees
td = TraceDiff(tree1, tree2)
td.generate_tracediff_report()  # Generates DataFrames, does NOT write files
td.print_tracediff_report_files('rprt_diff')  # Writes all reports to files in 'rprt_diff/'
```



**Output files:**
- `rprt_diff/merged_tree_output.txt`: Text visualization of the merged tree, showing matched and unmatched nodes.
- `rprt_diff/diff_stats.csv`: Detailed kernel and op statistics for each operation (see below for example and explanation).
- `rprt_diff/diff_stats_unique_args_summary.csv`: Aggregated summary statistics by op name and unique args.
- `rprt_diff/diff_stats_names_summary_df`: Aggregated summary stats by op name - giving top level summary.

---


## Output File Examples and Interpretation

#### diff_stats_names_summary_df.csv



**Example (first 5 rows):**

| name                        | row_count | kernel_time_trace1_sum_ms | kernel_time_trace2_sum_ms | diff_sum_ms  | abs_diff_sum_ms |
|-----------------------------|-----------|---------------------------|---------------------------|--------------|-----------------|
| aten::convolution_backward  | 736       | 541.957809                | 366.136090                | -175.821719  | 198.297619      |
| aten::_convolution          | 736       | 229.175081                | 157.807700                | -71.367381   | 85.731275       |
| aten::_batch_norm_impl_index| 448       | 129.995936                | 43.081600                 | -86.914335   | 86.914335       |
| aten::mm                    | 300       | 78.684444                 | 84.654093                 | 5.969649     | 11.982847       |
| FlashAttnFuncBackward       | 25        | 59.776381                 | 54.930648                 | -4.845733    | 4.845733        |



#### diff_stats.csv

This file contains detailed statistics for every op instance, including input shapes, types, kernel times, and kernel names. It is useful for fine-grained analysis and debugging.

#### diff_stats_unique_args_summary.csv

Midway between the detailed op wise view and the name summary this is a summary per argument of an operation. 

**How to use:**
- Drill down to individual op instances to investigate outliers or mismatches.
- Use the detailed input and kernel info to correlate with model code or trace events.

---

## Accessing DataFrames and UID Mapping

TraceDiff provides methods to access the detailed and summary DataFrames directly, as well as a `merged_uid_map` to cross-reference events between the two traces. This is useful for linking statistics or visualizations.

### Accessing DataFrames

```python
# After running td.generate_tracediff_report():
df_stats = td.diff_stats_df  # Detailed per-op DataFrame
df_unique_args_summary = td.diff_stats_unique_args_summary_df
df_name_summary = td.diff_stats_name_summary_df

if df is not None:
    print(df.head())
if df_unique_args_summary is not None:
    print(df_unique_args_summary.head())
if df_name_summary is not None:
    print(df_name_summary.head())
```

### UID Mapping Example

```python
# Get the corresponding UID in tree2 for a given UID in tree1
uid1 = next(iter(td.baseline.cpu_root_nodes))
uid2 = td.get_corresponding_uid(1, uid1)
if uid2 != -1:
    print(f"Tree1 UID {uid1} corresponds to Tree2 UID {uid2}")
else:
    print("No match found for this UID in tree2.")
```

---


## Use Cases

- **Performance Regression Analysis**: Quickly identify which operations or kernels have changed between two runs.
- **Debugging and Optimization**: Pinpoint new bottlenecks or regressions introduced by code or environment changes.
- **Cross-Trace Linking**: Map and compare specific events or kernels between two traces for deeper investigation.

---


## Notes

- TraceDiff is designed for PyTorch profiler JSON traces and requires TraceToTree objects as input.
- For more advanced usage, see the example notebook: `examples/trace_diff_example.ipynb`.
- Output folder and file names can be customized via the API.
- The API now separates report generation (`generate_tracediff_report`) from file output (`print_tracediff_report_files`).
- DataFrames are only available after running `generate_tracediff_report()`.
