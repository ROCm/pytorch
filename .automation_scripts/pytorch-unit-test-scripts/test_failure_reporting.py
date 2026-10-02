import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(__file__))

from auto_classify_skip_reasons import detect_columns
from generate_summary import (
    add_promoted_failure_stats,
    collect_log_failed_tests,
    write_markdown,
)


class FailureReportingTest(unittest.TestCase):
    def test_auto_classify_accepts_custom_primary_label(self):
        fields = [
            "test_file",
            "status_preview",
            "message_preview",
            "status_cuda",
            "message_cuda",
        ]
        self.assertEqual(
            detect_columns(fields),
            ("status_preview", "status_cuda", "message_preview"),
        )

    def test_log_crash_is_promoted_and_counted(self):
        base = {
            "arch": "preview",
            "platform": "rocm",
            "test_config": "inductor",
            "test_file": "inductor/test_origami",
            "job_shard": "1/2",
            "test_shard": "1/1",
            "reason": (
                "TestOrigami::"
                "test_origami_reduces_compile_work_vs_regular_max_autotune"
            ),
            "job_url": "https://github.com/pytorch/pytorch/actions/runs/1/job/2",
        }
        one_off = dict(
            base, status="FAILED", category="SEGFAULT")
        consistent = dict(
            base, status="FAILED_CONSISTENTLY",
            category="CONSISTENT_FAILURE")

        promoted = collect_log_failed_tests(
            [one_off, consistent], [], "preview")

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["status_preview"], "FAILED")
        self.assertIn("CONSISTENT_FAILURE", promoted[0]["error_message"])

        rows = [
            ("__section__", "TEST INDUCTOR"),
            ("PREVIEW", [10]),
            ("__section__", "OVERALL"),
            ("FAILED(preview)", [0]),
            ("TOTAL PREVIEW", [10]),
        ]
        add_promoted_failure_stats(rows, ["preview"], promoted, "preview")
        self.assertEqual(rows[1][1], [11])
        self.assertEqual(rows[3][1], [1])
        self.assertEqual(rows[4][1], [11])

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "summary.md")
            markdown = write_markdown(
                rows,
                ["preview"],
                output,
                failed_tests=promoted,
                s1_name="preview",
                s2_name="cuda",
                has_set2=False,
                log_failures=[one_off, consistent],
            )
        self.assertIn("### FAILED TESTS (1)", markdown)
        self.assertIn("CONSISTENT_FAILURE", markdown)
        self.assertNotIn("No failed tests found.", markdown)


if __name__ == "__main__":
    unittest.main()
