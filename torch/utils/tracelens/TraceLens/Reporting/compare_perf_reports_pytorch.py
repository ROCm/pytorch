#!/usr/bin/env python3
import argparse
import os
from typing import List, Sequence, Dict, Optional
import re
import pandas as pd
from openpyxl.utils import get_column_letter


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_list(obj) -> List[str]:
    return list(obj) if isinstance(obj, (list, tuple)) else [obj]


def load_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    """Read a sheet and verify that all key columns exist."""
    xls = pd.ExcelFile(path)
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"{path} has no sheet named '{sheet_name}'")
    df = pd.read_excel(xls, sheet_name=sheet_name)
    return df


def prefix_columns(df: pd.DataFrame, tag: str, keys: Sequence[str]) -> pd.DataFrame:
    """Prefix non-key columns with the report tag to keep them unique after merge."""
    keys = set(_ensure_list(keys))
    return df.rename(columns={c: f"{tag}::{c}" for c in df.columns if c not in keys})


def outer_merge(dfs: List[pd.DataFrame], keys: Sequence[str]) -> pd.DataFrame:
    """Perform an outer merge on a list of dataframes using the given keys."""
    keys = _ensure_list(keys)
    merged = dfs[0]
    for nxt in dfs[1:]:
        merged = pd.merge(merged, nxt, on=keys, how="outer")
    ordered = keys + [c for c in merged.columns if c not in keys]
    return merged[ordered]


def add_diff_cols(df: pd.DataFrame, tags: List[str], diff_cols: List[str] | str) -> pd.DataFrame:
    """
    Add *_diff and *_pct columns for:
      • given diff_col
      • total_direct_kernel_time_mean
    Diff = (variant_value - baseline_value)
    Pct  = 100 * diff / baseline_value
    """
    base_tag = tags[0]
    if isinstance(diff_cols, str):
        diff_cols = [diff_cols]

    for diff_col in diff_cols:
        base_col = f"{base_tag}::{diff_col}"
        for tag in tags[1:]:
            diff_col_name = f"{diff_col}__{tag}_diff"
            pct_col_name = f"{diff_col}__{tag}_pct"
            variant_col = f"{tag}::{diff_col}"
            df[diff_col_name] = df[variant_col] - df[base_col]
            denom = df[base_col].replace({0: pd.NA})
            df[pct_col_name] = 100 * (df[diff_col_name] / denom)
    return df


def build_df_dff(
        dfs: List[pd.DataFrame],
        list_report_tags: List[str],
        merge_keys: List[str],
        diff_cols: List[str] | str,
) -> pd.DataFrame:
    """
    Build a DataFrame with differences between multiple TraceLens reports.

    Parameters:
    - dfs: List of DataFrames, each loaded from a report's specified sheet.
    - list_report_tags: List of tags for each report, used for column naming.
    - merge_keys: List of column names to merge on (e.g., ['name'] for ops_summary).
    - diff_cols: List of column names to compute differences for, or a single string
      representing a single column (e.g., 'total_direct_kernel_time_mean').

    Returns:
    A DataFrame with merged data, difference columns, and percentage columns.
    """

    # 1. Prefix columns in each DataFrame with the report tag
    dfs = [
        prefix_columns(df, tag, merge_keys)
        for df, tag in zip(dfs, list_report_tags)
    ]
    # 2. Merge the DataFrames on the specified keys
    merged_df = outer_merge(dfs, merge_keys)
    # 3. Add diff and pct columns for the specified diff_col
    merged_df = add_diff_cols(merged_df, list_report_tags, diff_cols)
    # 4. Reorder columns: keys, diff cols, then all other metrics
    diff_cols = [col for col in merged_df.columns if re.match(r".*__.*_diff|.*__.*_pct", col)]
    ordered_cols = (
        merge_keys
        + diff_cols
        + [col for col in merged_df.columns if col not in merge_keys and col not in diff_cols]
    )
    
    return merged_df[ordered_cols]


def split_df_diff(
    name: str,
    df_diff: pd.DataFrame,
    tags: List[str],
    diff_col: str,
    sort_col: str,
    drop_other_tag_cols: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Returns three data-frames per variant tag (vs. baseline):

        <name>_intersect_<tag>
        <name>_only_baseline_<tag>
        <name>_only_variant_<tag>

    If `drop_other_tag_cols` is True, each frame keeps only:

        • key columns   (those without '::')
        • columns whose prefix matches *kept_tags*
        • *_diff / *_pct columns that refer only to *kept_tags*

    Parameters:
    - name: Base name for the resulting DataFrames.
    - df_diff: DataFrame containing the differences between reports.
    - tags: List of report tags, where the first tag is considered the baseline.
    - diff_col: The column name to compute differences for (e.g., 'total_direct_kernel_time_mean').
    - sort_col: The column name to sort the results by (e.g., 'total_direct_kernel_time_sum').
    - drop_other_tag_cols: If True, drop columns whose tag prefix is not in kept_tags.
    """
    baseline_tag = tags[0]
    results = {}

    # --- little utility ------------------------------------------------------
    def _strip_other_tags(df: pd.DataFrame, kept_tags: set[str]) -> pd.DataFrame:
        """
        Drop columns whose tag-prefix isn’t in kept_tags, plus all-NA cols,
        while preserving the original column order.
        """
        final_cols = []
        diff_pct_pattern = r"__(.+?)_(diff|pct)$"

        # Iterate through columns in their existing order to preserve it
        for col in df.columns:
            # Check if it's a key column (no '::' and not a diff/pct col)
            if "::" not in col and not re.search(diff_pct_pattern, col):
                final_cols.append(col)
                continue

            # Check if it's a data column (e.g., 'baseline::metric') with a kept tag
            if "::" in col:
                tag = col.split("::", 1)[0]
                if tag in kept_tags:
                    final_cols.append(col)
                continue

            # Check if it's a diff/pct column whose variant tag is a kept tag
            # This assumes the baseline is always implicitly part of the comparison
            if re.search(diff_pct_pattern, col):
                variant_tags = re.findall(r"__(.+?)_(?:diff|pct)$", col)
                if all(t in kept_tags for t in variant_tags):
                    final_cols.append(col)

        return df[final_cols].dropna(axis=1, how="all")
    
    # ------------------------------------------------------------------------
    for tag in tags[1:]:               # each non-baseline report
        intersect =  df_diff[f"{tag}::{diff_col}"].notna() & df_diff[f"{baseline_tag}::{diff_col}"].notna()
        base_only =  df_diff[f"{baseline_tag}::{diff_col}"].notna() & df_diff[f"{tag}::{diff_col}"].isna()
        var_only  =  df_diff[f"{tag}::{diff_col}"].notna() & df_diff[f"{baseline_tag}::{diff_col}"].isna()

        # 1) INTERSECT  – keep both tags
        df_i = df_diff.loc[intersect].sort_values(
            f"{baseline_tag}::{sort_col}", ascending=False, na_position="last"
        ).reset_index(drop=True)
        if drop_other_tag_cols:
            df_i = _strip_other_tags(df_i, {baseline_tag, tag})
        results[f"{name}_intersect_{tag}"] = df_i

        # 2) BASELINE-ONLY – drop variant’s tag columns
        df_b = df_diff.loc[base_only].sort_values(
            f"{baseline_tag}::{sort_col}", ascending=False, na_position="last"
        ).reset_index(drop=True)
        if drop_other_tag_cols:
            df_b = _strip_other_tags(df_b, {baseline_tag})
        results[f"{name}_only_baseline_{tag}"] = df_b

        # 3) VARIANT-ONLY – drop baseline’s tag columns
        df_v = df_diff.loc[var_only].sort_values(
            f"{tag}::{sort_col}", ascending=False, na_position="last"
        ).reset_index(drop=True)
        if drop_other_tag_cols:
            df_v = _strip_other_tags(df_v, {tag})
        results[f"{name}_only_variant_{tag}"] = df_v

    return results

def generate_compare_perf_reports_pytorch(
        reports: List[str], # List of paths to TraceLens reports
        output: str = "comparison.xlsx",
        names: List[str] = None,
        sheets: List[str] = ["all"],
) -> Dict[str, pd.DataFrame]:
    
    tags = (
        names
        if names
        else [os.path.splitext(os.path.basename(p))[0] for p in reports]
    )
    baseline_tag = tags[0]
    if len(set(tags)) != len(tags):
        raise ValueError("Tags must be unique – use --names to disambiguate.")
    results: dict[str, pd.DataFrame] = {}
    cols_to_hide_xl: dict[str, List[str]] = {}

    # ── GPU timeline ──────────────────────────────────────────────────────────
    if "gpu_timeline" in sheets or "all" in sheets:
        keys = ["type"]
        diff_col = "time ms"
        # Load the GPU timeline sheet from each report
        dfs = [
            load_sheet(path, sheet_name="gpu_timeline")
            for path in reports
        ]
        dtl = build_df_dff(
            dfs=dfs,
            list_report_tags=tags,
            merge_keys=keys,
            diff_cols=diff_col,
        )
        results["gpu_timeline"] = dtl

    # ── Ops summary ───────────────────────────────────────────────────────────
    if "ops_summary" in sheets or "all" in sheets:
        keys = ["name"]
        diff_cols = ["total_direct_kernel_time_ms", "Count"]
        cols_to_delete = ["total_direct_kernel_time_sum"]
        # Load the Ops summary sheet from each report
        dfs = [
            load_sheet(path, sheet_name="ops_summary")
            for path in reports
        ]

        # Delete columns that are not needed
        for i, df in enumerate(dfs):
            cols_to_delete = ["total_direct_kernel_time_sum"]
            if i > 0:
                cols_to_delete.append("Cumulative Percentage (%)")
            df.drop(columns=cols_to_delete, inplace=True, errors='ignore')
        
        ops = build_df_dff(
            dfs=dfs,
            list_report_tags=tags,
            merge_keys=keys,
            diff_cols=diff_cols
        )
        
        # sort by baseline tag's total_direct_kernel_time_ms
        sort_key = f"{baseline_tag}::total_direct_kernel_time_ms"
        
        ops = ops.sort_values(sort_key, ascending=False).reset_index(drop=True)
        
        results["ops_summary"] = ops

    # ── Ops ALL (split into 3 sheets) ─────────────────────────────────────────
    alias = ["ops_all", "ops_unique_args"] # different names for different versions of perf reports
    if "ops_all" in sheets or "all" in sheets:
        for sheet_name in alias:
            if sheet_name in pd.ExcelFile(reports[0]).sheet_names:
                ops_all_sheet = sheet_name
                break
        keys = [
            "name",
            "Input type",
            "Input Dims",
            "Input Strides",
            "Concrete Inputs",
        ]
        diff_cols = ["total_direct_kernel_time_sum", "total_direct_kernel_time_mean", "operation_count"]

        dfs = [
            load_sheet(path, sheet_name=ops_all_sheet)
            for path in reports
        ]

        opsA = build_df_dff(
            dfs=dfs,
            list_report_tags=tags,
            merge_keys=keys,
            diff_cols=diff_cols,
        )

        this_results = split_df_diff(
            name="ops_all",
            df_diff=opsA,
            tags=tags,
            diff_col=diff_cols[0],  # use the first diff_col for checking matches
            sort_col="total_direct_kernel_time_sum",
            drop_other_tag_cols=True,  # keep only keys and diff/pct cols for kept tags
        )
        results.update(this_results)

        for sheet_name in this_results.keys():
            cols_to_hide = [c for c in this_results[sheet_name].columns if c.endswith(
                ("kernel_names", "median", "std", "min", "max", "ex_UID")
            )]
            cols_to_hide_xl[sheet_name] = cols_to_hide

    # ── Roofline sheets (per-op) ──────────────────────────────────────────────
    if "roofline" in sheets or "all" in sheets:
        roofline_sheets = [
            "GEMM",
            "SDPA_fwd",
            "SDPA_bwd",
            "CONV_fwd",
            "CONV_bwd",
            "UnaryElementwise",
            "BinaryElementwise",
        ]
        roofline_short_names = {
            "GEMM": "GEMM",
            "SDPA_fwd": "SDPA_fwd",
            "SDPA_bwd": "SDPA_bwd",
            "CONV_fwd": "CONV_fwd",
            "CONV_bwd": "CONV_bwd",
            "UnaryElementwise": "un_eltwise",
            "BinaryElementwise": "bin_eltwise",
        }

        for sheet in roofline_sheets:
            
            dfs = [
                load_sheet(path=path, sheet_name=sheet)
                for path in reports
            ]

            # delete columns that are not needed for non-baseline reports
            # like GFLOPS_first, Data Moved (MB)_first as these are same for all
            cols_to_del_non_baseline = [
                "GFLOPS_first", "Data Moved (MB)_first", "FLOPS/Byte_first",
                "Input type_first", "Input Dims_first", "Input Strides_first",
                "Concrete Inputs_first"]
            for i, df in enumerate(dfs):
                if i > 0:
                    df.drop(columns=cols_to_del_non_baseline, inplace=True, errors='ignore')
                    
            # load the baseline report to get the merge keys
            df_roofline_ref = dfs[0]
            cond = lambda col: col.startswith("param:")
            merge_keys = ['name'] + [col for col in df_roofline_ref.columns if cond(col)]
            diff_cols = ['Kernel Time (µs)_sum', 'Kernel Time (µs)_mean', "name_count",
                         'TFLOPS/s_mean', 'TB/s_mean']
            # if any of dfs is empty, skip this sheet
            if any(df.empty for df in dfs):
                print(f"Skipping roofline sheet '{sheet}' because one of the reports is empty.")
                continue

            # Load the roofline sheet for each report
            roofline_diff = build_df_dff(
                dfs=dfs,
                list_report_tags=tags,
                merge_keys=merge_keys,
                diff_cols=diff_cols,
            )
            this_results = split_df_diff(
                    name=roofline_short_names[sheet],
                    df_diff=roofline_diff,
                    tags=tags,
                    diff_col=diff_cols[0],  # use the first diff_col for checking matches
                    sort_col="Kernel Time (µs)_sum",
            )
            
            results.update(this_results)

            for sheet_name in this_results.keys():
                cols_to_hide = [c for c in this_results[sheet_name].columns if c.endswith(
                    ("kernel_names_first", "UID", "median", "std", "min", "max",
                     "Input type_first", "Input Dims_first", "Input Strides_first",
                     "Concrete Inputs_first")
                )]
                cols_to_hide_xl[sheet_name] = cols_to_hide

    # ── Write workbook ────────────────────────────────────────────────────────
    with pd.ExcelWriter(output, engine='openpyxl') as xls:
        for sheet_name, df in results.items():
            # if df is empty, skip writing it
            if df.empty:
                print(f"Skipping empty sheet '{sheet_name}'")
                continue
            df.to_excel(xls, sheet_name=sheet_name[:31], index=False)  # Excel 31-char limit
            for col in cols_to_hide_xl.get(sheet_name, []):
                col_idx = df.columns.get_loc(col) + 1
                col_letter = get_column_letter(col_idx)
                worksheet = xls.sheets[sheet_name[:31]]
                worksheet.column_dimensions[col_letter].hidden = True
            print(f"Wrote sheet '{sheet_name}' with {len(df)} rows × {len(df.columns)} columns")
    
    return results


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="TraceLens Excel reports (.xlsx)")
    parser.add_argument(
        "-o", "--output", default="comparison.xlsx", help="Output file name"
    )
    parser.add_argument(
        "--names", nargs="*", help="Optional tags for each report (must match count)"
    )
    parser.add_argument(
        "--sheets",
        nargs="+",
        choices=(
            "gpu_timeline",
            "ops_summary",
            "ops_all",
            "roofline",
            "all",
        ),
        default=["all"],
        help="Which sheet groups to process. Can be one or more.",
    )
    args = parser.parse_args()

    if len(args.reports) < 2:
        parser.error("Need at least two report files")
    if args.names and len(args.names) != len(args.reports):
        parser.error("--names count must equal number of reports")

    generate_compare_perf_reports_pytorch(
        reports=args.reports,
        output=args.output,
        names=args.names,
        sheets=args.sheets,
    )

if __name__ == "__main__":
    main()
