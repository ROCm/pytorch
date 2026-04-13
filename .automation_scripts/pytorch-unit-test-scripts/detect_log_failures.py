#!/usr/bin/env python3
"""Scan CI log files (.txt) for test failures not captured in XML reports.

Tests that timeout (exit code 124), crash (SIGIOT, SIGSEGV, Fatal Python error),
or are killed (SIGKILL, OOM) never produce JUnit XML output. This script detects
those failures from the raw log files and outputs a CSV/summary.

Usage:
    python detect_log_failures.py --logs-dir <folder> [--output <path.csv>]
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


RE_RUNNING = re.compile(
    r"Running (?P<test_file>\S+) (?P<shard>\d+)/(?P<total>\d+) \.\.\."
)
RE_SUCCESS = re.compile(
    r"(?P<test_file>\S+) (?P<shard>\d+)/(?P<total>\d+) was successful"
)
RE_FAILED = re.compile(
    r"(?P<test_file>\S+) (?P<shard>\d+)/(?P<total>\d+) failed!(?P<reason>.*)"
)
RE_EXIT_CODE = re.compile(r"Got exit code (?P<code>\d+)")
RE_TIMEOUT = re.compile(r"Command took >(\d+)min, returning 124")
RE_FAILED_CONSISTENTLY = re.compile(
    r"FAILED CONSISTENTLY: (?P<test_path>\S+)"
)
RE_STEPCURRENT = re.compile(
    r"stepcurrent:.*Running only (?:test/)?(?P<test_path>\S+)"
)
RE_INDIVIDUAL_TEST = re.compile(
    r"(?P<test_path>\S+\.py::(?P<cls>\w+)::(?P<method>\w+))"
)

CRASH_PATTERNS = [
    (re.compile(r"Segmentation fault", re.IGNORECASE), "SEGFAULT"),
    (re.compile(r"SIGSEGV"), "SIGSEGV"),
    (re.compile(r"SIGIOT"), "SIGIOT"),
    (re.compile(r"SIGABRT"), "SIGABRT"),
    (re.compile(r"SIGKILL"), "SIGKILL"),
    (re.compile(r"Fatal Python error", re.IGNORECASE), "FATAL_PYTHON"),
    (re.compile(r"core dumped", re.IGNORECASE), "CORE_DUMP"),
    (re.compile(r"Aborted \(core dumped\)", re.IGNORECASE), "ABORTED"),
    (re.compile(r"torch\.cuda\.OutOfMemoryError"), "CUDA_OOM"),
    (re.compile(r"std::bad_alloc"), "BAD_ALLOC"),
]

LOG_FILE_MAP = {
    "rocm": ("rocm", "default"),
    "rocm_dist": ("rocm", "distributed"),
    "rocm_inductor": ("rocm", "inductor"),
    "cuda": ("cuda", "default"),
    "cuda_dist": ("cuda", "distributed"),
    "cuda_inductor": ("cuda", "inductor"),
    "baseline": ("baseline", "default"),
}


def classify_log_file(filename):
    """Return (platform, test_config, shard_num) from a log filename like rocm3.txt."""
    stem = Path(filename).stem
    for prefix, (platform, test_config) in sorted(LOG_FILE_MAP.items(), key=lambda x: -len(x[0])):
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            if remainder.isdigit():
                return platform, test_config, int(remainder)
    return None, None, None


RE_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*")


def parse_log_file(filepath):
    """Parse a single log file and return test file results and consistent failures."""
    results = {}
    current_test = None
    last_failed_test = None
    consistent_failures = []

    with open(filepath, "r", errors="replace") as f:
        for line in f:
            # Lightweight tracking of individual pytest test lines.
            # These are very frequent (~37% of lines) so we extract the
            # test name directly without timestamp stripping.
            if ".py::" in line:
                m_ind = RE_INDIVIDUAL_TEST.search(line)
                if m_ind:
                    active = current_test or last_failed_test
                    if active and active in results:
                        # Only update if the pytest path belongs to this shard's test file,
                        # otherwise rerun output from earlier shards contaminates later ones.
                        shard_file = results[active]["test_file"]
                        if shard_file + ".py" in m_ind.group("test_path"):
                            results[active]["last_test"] = f"{m_ind.group('cls')}::{m_ind.group('method')}"

            if " ... [" not in line and "was successful" not in line \
               and "failed!" not in line and "Got exit code" not in line \
               and "returning 124" not in line and "FAILED CONSISTENTLY" not in line \
               and "Retrying" not in line \
               and "Segmentation fault" not in line and "SIGIOT" not in line \
               and "SIGSEGV" not in line and "SIGABRT" not in line \
               and "SIGKILL" not in line \
               and "Fatal Python error" not in line and "core dumped" not in line \
               and "Aborted (core dumped)" not in line \
               and "OutOfMemoryError" not in line \
               and "bad_alloc" not in line \
               and "stepcurrent" not in line:
                continue

            stripped = RE_TIMESTAMP.sub("", line).rstrip()

            m = RE_RUNNING.search(stripped)
            if m:
                key = f"{m.group('test_file')} {m.group('shard')}/{m.group('total')}"
                current_test = key
                if key not in results:
                    results[key] = {
                        "test_file": m.group("test_file"),
                        "shard": int(m.group("shard")),
                        "total": int(m.group("total")),
                        "status": "RUNNING",
                        "reason": "",
                        "exit_codes": [],
                        "crashes": [],
                        "crash_tests": [],
                        "last_test": "",
                    }
                continue

            m = RE_SUCCESS.search(stripped)
            if m:
                key = f"{m.group('test_file')} {m.group('shard')}/{m.group('total')}"
                if key in results:
                    results[key]["status"] = "PASSED"
                current_test = None
                last_failed_test = None
                continue

            m = RE_FAILED.search(stripped)
            if m:
                key = f"{m.group('test_file')} {m.group('shard')}/{m.group('total')}"
                reason = m.group("reason").strip()
                if key in results:
                    results[key]["status"] = "FAILED"
                    if reason:
                        results[key]["reason"] = reason
                last_failed_test = key
                current_test = key
                continue

            active = current_test or last_failed_test

            # Track stepcurrent rerun lines — identifies crash-causing test
            m = RE_STEPCURRENT.search(stripped)
            if m:
                test_path = m.group("test_path")
                parts = test_path.split("::")
                if len(parts) >= 3:
                    crash_id = f"{parts[1]}::{parts[2]}"
                elif len(parts) == 2:
                    crash_id = parts[1]
                else:
                    crash_id = None
                if crash_id and active and active in results:
                    shard_file = results[active]["test_file"]
                    if shard_file in test_path:
                        if crash_id not in results[active]["crash_tests"]:
                            results[active]["crash_tests"].append(crash_id)
                continue

            # Track individual pytest test lines for last-running-test context
            m_ind = RE_INDIVIDUAL_TEST.search(stripped)
            if m_ind and active and active in results:
                cls = m_ind.group("cls")
                method = m_ind.group("method")
                results[active]["last_test"] = f"{cls}::{method}"

            m = RE_EXIT_CODE.search(stripped)
            if m:
                code = int(m.group("code"))
                if active and active in results:
                    results[active]["exit_codes"].append(code)

            m = RE_TIMEOUT.search(stripped)
            if m and active and active in results:
                if "TIMEOUT" not in results[active]["crashes"]:
                    results[active]["crashes"].append("TIMEOUT")

            m = RE_FAILED_CONSISTENTLY.search(stripped)
            if m:
                consistent_failures.append(m.group("test_path"))

            if active and active in results:
                for pattern, label in CRASH_PATTERNS:
                    if pattern.search(stripped):
                        if label not in results[active]["crashes"]:
                            results[active]["crashes"].append(label)

    return results, consistent_failures


def scan_logs(logs_dir):
    """Scan all log files and return all non-passing test file results."""
    all_failures = []

    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".txt"):
            continue

        platform, test_config, shard_num = classify_log_file(fname)
        if platform is None:
            continue

        filepath = os.path.join(logs_dir, fname)
        results, consistent_failures = parse_log_file(filepath)

        for key, info in results.items():
            if info["status"] == "PASSED":
                continue

            categories = []
            if 124 in info["exit_codes"] or "TIMEOUT" in info["crashes"]:
                categories.append("TIMEOUT")
            for c in info["crashes"]:
                if c != "TIMEOUT":
                    categories.append(c)
            if info["status"] == "FAILED" and not categories:
                categories.append("FAILED")
            if info["status"] == "RUNNING" and not categories:
                categories.append("INCOMPLETE")

            if not categories:
                continue
            # Skip tests stuck in RUNNING with no evidence of failure —
            # these are typically from multi-shard logs where a different
            # shard's "Running ..." line appeared but the result was elsewhere.
            if info["status"] == "RUNNING" and categories == ["INCOMPLETE"]:
                continue

            reason = info["reason"]
            # Populate reason with identified crash/timeout test name
            crash_tests = info.get("crash_tests", [])
            last_test = info.get("last_test", "")
            identified_test = ""
            if crash_tests:
                identified_test = crash_tests[0]
            elif last_test:
                identified_test = last_test

            if identified_test and "::" in identified_test:
                if not reason:
                    reason = identified_test
                elif "::" not in reason:
                    reason = f"{identified_test} | {reason}"

            all_failures.append({
                "log_file": fname,
                "platform": platform,
                "test_config": test_config,
                "test_file": info["test_file"],
                "shard": f"{info['shard']}/{info['total']}",
                "status": info["status"],
                "category": "+".join(categories),
                "reason": reason,
                "exit_codes": ",".join(str(c) for c in info["exit_codes"]),
            })

        for test_path in consistent_failures:
            parts = test_path.split("::")
            file_part = parts[0].replace("test/", "").replace(".py", "")
            test_class = parts[1] if len(parts) > 1 else ""
            test_name = parts[2] if len(parts) > 2 else ""

            all_failures.append({
                "log_file": fname,
                "platform": platform,
                "test_config": test_config,
                "test_file": file_part,
                "shard": "",
                "status": "FAILED_CONSISTENTLY",
                "category": "CONSISTENT_FAILURE",
                "reason": f"{test_class}::{test_name}" if test_class else "",
                "exit_codes": "",
            })

    return all_failures


def write_csv_report(failures, output_path):
    fieldnames = [
        "log_file", "platform", "test_config", "test_file", "shard",
        "status", "category", "reason", "exit_codes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)
    print(f"Log failure report: {output_path} ({len(failures)} entries)")


def print_summary(failures):
    if not failures:
        print("No log-based failures detected.")
        return

    by_category = defaultdict(list)
    for f in failures:
        by_category[f["category"]].append(f)

    print(f"\n{'='*60}")
    print("LOG FAILURE DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total failures detected: {len(failures)}")
    print()

    for cat, items in sorted(by_category.items()):
        print(f"  {cat}: {len(items)}")
        for item in items:
            print(f"    - {item['test_file']} ({item['platform']}/{item['test_config']}) [{item['log_file']}]")
            if item["reason"]:
                print(f"      Reason: {item['reason'][:120]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detect test failures from CI log files not captured in XML reports"
    )
    parser.add_argument(
        "--logs-dir", required=True,
        help="Directory containing .txt log files"
    )
    parser.add_argument(
        "--output", default="log_failures.csv",
        help="Output CSV path (default: log_failures.csv)"
    )
    args = parser.parse_args()

    failures = scan_logs(args.logs_dir)
    print_summary(failures)
    write_csv_report(failures, args.output)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
