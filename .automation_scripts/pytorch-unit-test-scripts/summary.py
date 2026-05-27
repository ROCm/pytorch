#!/usr/bin/env python3
"""Convert JUnit XML test reports to a JSON ingest record for FrameworkWeb.

Output schema matches the ingest endpoint described in
https://github.com/ROCm/frameworks-internal/tree/ingesting_ut_tracker/pytorch-unit-test-scripts/ut_results_ingestion/FrameworkWeb
(see utt/models.py).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


class Status:
    PASSED = "passed"
    SKIPPED = "skipped"
    FAILED = "failed"
    MISSED = "missed"
    XFAILED = "xfailed"
    ERROR = "error"


def _classify(testcase):
    skipped = testcase.find("skipped")
    if skipped is not None:
        if skipped.attrib.get("type") == "pytest.xfail":
            return Status.XFAILED
        return Status.SKIPPED
    if testcase.find("failure") is not None:
        return Status.FAILED
    if testcase.find("error") is not None:
        return Status.ERROR
    return Status.PASSED


def _collect(node, results):
    for child in node:
        if child.tag == "testcase":
            try:
                results.append({
                    "file": child.attrib["file"],
                    "classname": child.attrib["classname"],
                    "name": child.attrib["name"],
                    "time": child.attrib["time"],
                    "status": _classify(child),
                })
            except KeyError:
                # Skip testcases missing required attributes (e.g., aggregated
                # entries with no file/classname/name/time).
                pass
        _collect(child, results)


def import_dirs(input_dirs):
    results = []
    for input_dir in input_dirs:
        before = len(results)
        for xml_path in Path(input_dir).rglob("*.xml"):
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError as exc:
                print(f"WARNING: skipping malformed XML {xml_path}: {exc}")
                continue
            _collect(tree.getroot(), results)
        print(f"Collected {len(results) - before} test results from {input_dir}")
    return results


def build_info(args):
    return {
        "url": args.build_url or "",
        "branch": args.branch_name or "",
        "commit": args.commit_sha or "",
        "gfxArch": args.gfx_arch or "",
        "repoOwner": args.repo_owner or "",
        "rocmVersion": args.rocm_version or "",
        "pytorchVersion": args.pytorch_version or "",
        "testConfig": args.test_config or "",
        "buildTimestamp": (
            args.build_timestamp
            if args.build_timestamp
            else str(datetime.now(tz=timezone.utc))
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert XML test reports to a FrameworkWeb JSON ingest record"
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dirs",
        type=str,
        action="append",
        required=True,
        help="Directory containing JUnit XML files (recursive). May be repeated.",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        type=str,
        required=True,
        help="Path to write JSON output",
    )
    parser.add_argument("--build_url", type=str, default="", help="CI build URL")
    parser.add_argument("--branch_name", type=str, default="", help="Source branch name")
    parser.add_argument("--commit_sha", type=str, default="", help="Commit SHA under test")
    parser.add_argument("--gfx_arch", type=str, default="", help="GPU architecture (e.g. gfx942)")
    parser.add_argument("--repo_owner", type=str, default="", help="Repository owner")
    parser.add_argument("--rocm_version", type=str, default="", help="ROCm version")
    parser.add_argument("--pytorch_version", type=str, default="", help="PyTorch version")
    parser.add_argument("--test_config", type=str, default="", help="Test config (default, distributed, inductor)")
    parser.add_argument("--build_timestamp", type=str, default="", help="Override build timestamp (default: now UTC)")
    return parser.parse_args()


def main():
    args = parse_args()
    data = {
        "build": build_info(args),
        "results": import_dirs(args.input_dirs),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
