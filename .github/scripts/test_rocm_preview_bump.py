#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path

from rocm_preview_bump import evaluate, load_source, render_pr_body


CURRENT_PIN = "10.2.0a20261015"


def payload(*, failure_count: object = 9) -> dict[str, object]:
    return {
        "schema_version": "2.2",
        "rocm_version": "10.2.0a20261016",
        "build_date": "20261016",
        "pipelines": {
            "pytorch": {
                "test_full": {
                    "linux": {
                        "gfx94X-dcgpu": {
                            "variants": [
                                {
                                    "py": "3.12",
                                    "torch": "release/2.14",
                                    "status": "success",
                                    "test_results_complete": True,
                                    "test_failure_count": 0,
                                },
                                {
                                    "py": "3.12",
                                    "torch": "nightly",
                                    "status": "failure",
                                    "test_results_complete": True,
                                    "test_failure_count": failure_count,
                                },
                                {
                                    "py": "3.13",
                                    "torch": "nightly",
                                    "status": "success",
                                    "test_results_complete": True,
                                    "test_failure_count": 20,
                                },
                            ]
                        }
                    }
                }
            }
        },
    }


class TestEvaluate(unittest.TestCase):
    def test_failure_threshold_is_strict(self) -> None:
        for count, eligible in ((9, True), (10, False), (20, False)):
            with self.subTest(count=count):
                self.assertEqual(
                    evaluate(payload(failure_count=count), CURRENT_PIN).eligible,
                    eligible,
                )

    def test_incomplete_results_are_ineligible(self) -> None:
        data = payload()
        variant = data["pipelines"]["pytorch"]["test_full"]["linux"]["gfx94X-dcgpu"][
            "variants"
        ][1]
        variant["test_results_complete"] = False
        self.assertFalse(evaluate(data, CURRENT_PIN).eligible)

    def test_malformed_and_negative_failure_counts_are_ineligible(self) -> None:
        for count in (-1, "9", 1.5, True, None):
            with self.subTest(count=count):
                self.assertFalse(
                    evaluate(payload(failure_count=count), CURRENT_PIN).eligible
                )

    def test_stale_version_or_build_date_is_ineligible(self) -> None:
        cases = (
            ("10.2.0a20261015", "20261016"),
            ("10.2.0a20261016", "20261015"),
            ("10.1.0a20261017", "20261017"),
        )
        for version, build_date in cases:
            with self.subTest(version=version, build_date=build_date):
                data = payload()
                data["rocm_version"] = version
                data["build_date"] = build_date
                self.assertFalse(evaluate(data, CURRENT_PIN).eligible)

    def test_default_variant_selection_ignores_other_variants(self) -> None:
        self.assertTrue(evaluate(payload(), CURRENT_PIN).eligible)

    def test_variant_selection_can_be_overridden(self) -> None:
        data = payload()
        result = evaluate(
            data,
            CURRENT_PIN,
            python_version="3.13",
            torch_variant="nightly",
            threshold=21,
        )
        self.assertTrue(result.eligible)
        self.assertEqual(result.test_failure_count, 20)

    def test_missing_or_duplicate_variant_is_ineligible(self) -> None:
        data = payload()
        variants = data["pipelines"]["pytorch"]["test_full"]["linux"]["gfx94X-dcgpu"][
            "variants"
        ]
        variants.append(copy.deepcopy(variants[1]))
        self.assertFalse(evaluate(data, CURRENT_PIN).eligible)
        self.assertFalse(
            evaluate(payload(), CURRENT_PIN, torch_variant="missing").eligible
        )

    def test_requires_schema_2_2_and_terminal_status(self) -> None:
        data = payload()
        data["schema_version"] = "2.1"
        self.assertFalse(evaluate(data, CURRENT_PIN).eligible)
        data["schema_version"] = "2.2"
        data["pipelines"]["pytorch"]["test_full"]["linux"]["gfx94X-dcgpu"]["variants"][
            1
        ]["status"] = "in_progress"
        self.assertFalse(evaluate(data, CURRENT_PIN).eligible)

    def test_local_source_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "latest.json"
            source.write_text(json.dumps(payload()), encoding="utf-8")
            self.assertEqual(load_source(str(source)), payload())
            self.assertEqual(load_source(source.as_uri()), payload())

    def test_github_contents_api_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "response.json"
            encoded = base64.b64encode(json.dumps(payload()).encode()).decode()
            source.write_text(
                json.dumps({"encoding": "base64", "content": encoded}),
                encoding="utf-8",
            )
            self.assertEqual(load_source(str(source)), payload())


class TestRenderPrBody(unittest.TestCase):
    TEMPLATE = """Intro

## Linked issue or supporting maintainer

placeholder

## BC-breaking?

placeholder

## Summary (human written only)

Keep this.

## Agent details (if applicable)

placeholder
"""

    def test_renders_required_sections_and_marker(self) -> None:
        body = render_pr_body(
            self.TEMPLATE,
            rocm_version="10.2.0a20261016",
            build_date="20261016",
            issue="123",
        )
        self.assertIn("Part of #123", body)
        self.assertIn("## BC-breaking?\n\nNo", body)
        self.assertIn("<!-- rocm-preview-nightly-bump -->", body)
        self.assertIn("AIPYTORCH-1525", body)
        self.assertNotIn("atlassian", body)
        self.assertIn("Keep this.", body)

    def test_requires_exactly_one_linkage_option(self) -> None:
        for issue, maintainer in (("", ""), ("123", "owner")):
            with self.subTest(issue=issue, maintainer=maintainer):
                with self.assertRaises(ValueError):
                    render_pr_body(
                        self.TEMPLATE,
                        rocm_version="10.2.0a20261016",
                        build_date="20261016",
                        issue=issue,
                        maintainer=maintainer,
                    )


if __name__ == "__main__":
    unittest.main()
