#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_SOURCE = "https://api.github.com/repos/ROCm/Quartz/contents/nightly/latest.json"
MINIMUM_SCHEMA = (2, 2)
TERMINAL_STATUSES = {"success", "failure", "cancelled", "skipped", "timed_out"}


@dataclass(frozen=True)
class Evaluation:
    eligible: bool
    reason: str
    rocm_version: str = ""
    build_date: str = ""
    test_failure_count: int | None = None


def load_source(source: str) -> dict[str, Any]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https", "file"}:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "rocm-preview-bump-bot",
        }
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        request = Request(source, headers=headers)
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    elif parsed.scheme:
        raise ValueError(f"unsupported source scheme: {parsed.scheme}")
    else:
        payload = Path(source).read_bytes()

    data = json.loads(payload)
    if (
        isinstance(data, dict)
        and data.get("encoding") == "base64"
        and isinstance(data.get("content"), str)
    ):
        data = json.loads(base64.b64decode(data["content"]))
    if not isinstance(data, dict):
        raise ValueError("Quartz payload must be a JSON object")
    return data


def _schema_version(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)*", value):
        return None
    return tuple(int(part) for part in value.split("."))


def _version_key(value: str) -> tuple[int, ...] | None:
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts) if parts else None


def _build_date(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"20\d{6}", value):
        return None
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return None
    return value


def _pin_date(pin: str) -> str | None:
    matches = re.findall(r"(?<!\d)(20\d{6})(?!\d)", pin)
    return matches[-1] if matches else None


def _ineligible(reason: str, data: dict[str, Any]) -> Evaluation:
    return Evaluation(
        eligible=False,
        reason=reason,
        rocm_version=str(data.get("rocm_version", "")),
        build_date=str(data.get("build_date", "")),
    )


def evaluate(
    data: dict[str, Any],
    current_pin: str,
    *,
    threshold: int = 10,
    framework: str = "pytorch",
    pipeline: str = "test_full",
    platform: str = "linux",
    target: str = "gfx94X-dcgpu",
    torch_variant: str = "nightly",
    python_version: str = "3.12",
) -> Evaluation:
    schema = _schema_version(data.get("schema_version"))
    if (
        schema is None
        or len(schema) < 2
        or schema[0] != MINIMUM_SCHEMA[0]
        or schema[1] < MINIMUM_SCHEMA[1]
    ):
        return _ineligible("Quartz schema must be compatible with 2.2", data)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        return _ineligible("threshold must be a positive integer", data)

    try:
        target_data = data["pipelines"][framework][pipeline][platform][target]
        variants = target_data["variants"]
    except (KeyError, TypeError):
        return _ineligible("target pipeline is missing", data)
    if not isinstance(variants, list):
        return _ineligible("target variants must be a list", data)

    matches = [
        variant
        for variant in variants
        if isinstance(variant, dict)
        and variant.get("torch") == torch_variant
        and str(variant.get("py")) == python_version
    ]
    if len(matches) != 1:
        return _ineligible(
            f"expected exactly one {torch_variant}/py{python_version} variant", data
        )

    variant = matches[0]
    if variant.get("status") not in TERMINAL_STATUSES:
        return _ineligible("selected variant is not terminal", data)
    if variant.get("test_results_complete") is not True:
        return _ineligible("selected variant test results are incomplete", data)

    failure_count = variant.get("test_failure_count")
    if (
        isinstance(failure_count, bool)
        or not isinstance(failure_count, int)
        or failure_count < 0
    ):
        return _ineligible("test_failure_count must be a nonnegative integer", data)
    if failure_count >= threshold:
        return Evaluation(
            False,
            f"test_failure_count {failure_count} is not below {threshold}",
            str(data.get("rocm_version", "")),
            str(data.get("build_date", "")),
            failure_count,
        )

    rocm_version = data.get("rocm_version")
    build_date = _build_date(data.get("build_date"))
    pin = current_pin.strip()
    candidate_key = (
        _version_key(rocm_version) if isinstance(rocm_version, str) else None
    )
    pin_key = _version_key(pin)
    pin_date = _pin_date(pin)
    if not rocm_version or candidate_key is None or pin_key is None:
        return _ineligible("ROCm version or current pin is malformed", data)
    if build_date is None or pin_date is None:
        return _ineligible("Quartz or pin build date is malformed", data)
    if candidate_key <= pin_key or build_date <= pin_date:
        return _ineligible(
            "Quartz ROCm version/build date is not newer than the pin", data
        )

    return Evaluation(
        True,
        "eligible",
        rocm_version,
        build_date,
        failure_count,
    )


def render_pr_body(
    template: str,
    *,
    rocm_version: str,
    build_date: str,
    issue: str = "",
    maintainer: str = "",
) -> str:
    if bool(issue) == bool(maintainer):
        raise ValueError(
            "configure exactly one actionable issue or supporting maintainer"
        )
    linked = f"Part of #{issue}" if issue else f"Supported by @{maintainer.lstrip('@')}"
    key = f"{rocm_version}+{build_date}"
    agent_details = f"""<!-- rocm-preview-nightly-bump -->
<!-- rocm-preview-key: {key} -->
### Problem
The ROCm preview pin must follow a completed, low-failure Quartz nightly.

### Changes
Update `.ci/docker/ci_commit_pins/rocm-preview.txt` to `{rocm_version}`.

### Validation
Quartz schema 2.2 eligibility passed for build `{build_date}` with fewer than the configured failure threshold.

Tracked in AIPYTORCH-1525."""
    body = _replace_section(template, "Linked issue or supporting maintainer", linked)
    body = _replace_section(body, "BC-breaking?", "No")
    return _replace_section(body, "Agent details (if applicable)", agent_details)


def _replace_section(template: str, heading: str, content: str) -> str:
    pattern = re.compile(rf"(?ms)^## {re.escape(heading)}\s*\n.*?(?=^## |\Z)")
    replacement = f"## {heading}\n\n{content}\n\n"
    result, count = pattern.subn(replacement, template, count=1)
    if count != 1:
        raise ValueError(f"template is missing section: {heading}")
    return result


def _write_github_output(path: str, result: Evaluation) -> None:
    values = asdict(result)
    with Path(path).open("a", encoding="utf-8") as output:
        for key, value in values.items():
            rendered = (
                ""
                if value is None
                else str(value).lower()
                if isinstance(value, bool)
                else str(value)
            )
            output.write(f"{key}={rendered}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--source", default=DEFAULT_SOURCE)
    evaluate_parser.add_argument("--pin-file", required=True)
    evaluate_parser.add_argument("--threshold", type=int, default=10)
    evaluate_parser.add_argument("--framework", default="pytorch")
    evaluate_parser.add_argument("--pipeline", default="test_full")
    evaluate_parser.add_argument("--platform", default="linux")
    evaluate_parser.add_argument("--target", default="gfx94X-dcgpu")
    evaluate_parser.add_argument("--torch-variant", default="nightly")
    evaluate_parser.add_argument("--python-version", default="3.12")
    evaluate_parser.add_argument("--github-output")

    render_parser = subparsers.add_parser("render-pr-body")
    render_parser.add_argument("--template", required=True)
    render_parser.add_argument("--output", required=True)
    render_parser.add_argument("--rocm-version", required=True)
    render_parser.add_argument("--build-date", required=True)
    render_parser.add_argument("--issue", default="")
    render_parser.add_argument("--maintainer", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "render-pr-body":
        template = Path(args.template).read_text(encoding="utf-8")
        body = render_pr_body(
            template,
            rocm_version=args.rocm_version,
            build_date=args.build_date,
            issue=args.issue,
            maintainer=args.maintainer,
        )
        Path(args.output).write_text(body, encoding="utf-8")
        return 0

    data = load_source(args.source)
    result = evaluate(
        data,
        Path(args.pin_file).read_text(encoding="utf-8"),
        threshold=args.threshold,
        framework=args.framework,
        pipeline=args.pipeline,
        platform=args.platform,
        target=args.target,
        torch_variant=args.torch_variant,
        python_version=args.python_version,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    if args.github_output:
        _write_github_output(args.github_output, result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
