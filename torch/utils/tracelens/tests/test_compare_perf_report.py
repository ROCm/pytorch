import os
import subprocess

import pandas as pd
import pytest
from pandas.api.types import is_float_dtype


def compare_cols(df_test, df_ref, cols, tol=1e-6):
    """Compare columns in two dataframes, skipping rows where ref is None/NaN."""
    diff_cols = []
    for col in cols:
        valid_mask = df_ref[col].notna()
        if not valid_mask.any():
            continue
        ref_col = df_ref.loc[valid_mask, col]
        test_col = df_test.loc[df_test.index.intersection(ref_col.index), col]
        test_col, ref_col = test_col.align(ref_col, join="right")
        if is_float_dtype(df_test[col]):
            diff = test_col - ref_col
            if not diff.abs().max() < tol:
                diff_cols.append(col)
        else:
            if not (test_col == ref_col).all():
                diff_cols.append(col)
    return diff_cols


def generate_perf_report(profile_path, report_path):
    script_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "TraceLens",
            "Reporting",
            "generate_perf_report_pytorch.py",
        )
    )
    cmd = [
        "python3",
        script_path,
        "--profile_json_path",
        profile_path,
        "--output_xlsx_path",
        report_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running command: {result.stderr}")
    return True


def find_test_cases(ref_root):
    """
    Recursively find all .json.gz/.xlsx pairs in ref_root. The .xlsx file must have the same basename as the .json.gz, but with _perf_report.xlsx.
    Returns a list of tuples: (profile_path, report_filename)
    """
    test_cases = []
    for dirpath, _, filenames in os.walk(ref_root):
        gz_files = [f for f in filenames if f.endswith(".json.gz")]
        for gz in gz_files:
            base = gz[:-8]  # remove .json.gz
            xlsx = base + "_perf_report.xlsx"
            xlsx_path = os.path.join(dirpath, xlsx)
            if os.path.exists(xlsx_path):
                test_cases.append((dirpath, gz, xlsx))
    return test_cases


@pytest.mark.parametrize("dirpath,gz,report_filename", find_test_cases("tests/traces"))
def test_compare_perf_report(dirpath, gz, report_filename, tol=1e-6):
    """
    For each .gz/.xlsx pair, decompress the .gz, generate a report, and compare to the reference .xlsx.
    """
    import shutil

    ref_root = dirpath
    profile_path = os.path.join(dirpath, gz)
    ref_report_path = os.path.join(dirpath, report_filename)
    # Generate a temp output directory for this test
    fn_root = os.path.join(dirpath, "pytest_reports")
    os.makedirs(fn_root, exist_ok=True)
    # Decompress .gz to .json (without deleting .gz)
    jsonfile = profile_path[:-3]
    try:
        subprocess.run(["gunzip", "-kf", profile_path], check=True)
        # Generate report
        fn_report_path = os.path.join(fn_root, report_filename)
        generate_perf_report(jsonfile, fn_report_path)
        # Compare sheets
        sheets = pd.ExcelFile(ref_report_path).sheet_names
        cols_ignore = [
            "Non-Data-Mov TFLOPS/s_mean",
            "Non-Data-Mov Kernel Time (µs)_sum",
            "Non-Data-Mov Kernel Time (µs)_mean",
            "kernel_details_summary",
            "trunc_kernel_details",
            "kernel_details__summarize_kernel_stats",
        ]
        for sheet in sheets:
            df_ref = pd.read_excel(ref_report_path, sheet_name=sheet)
            df_fn = pd.read_excel(fn_report_path, sheet_name=sheet)
            if df_ref.empty:
                continue
            cols = [col for col in df_ref.columns if col not in cols_ignore]
            diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
            assert (
                not diff_cols
            ), f"Sheet {sheet}: {diff_cols} are different in {profile_path}"
    finally:
        # Cleanup: remove generated .json and report
        if os.path.exists(jsonfile):
            os.remove(jsonfile)
        if os.path.exists(fn_root):
            shutil.rmtree(fn_root)
