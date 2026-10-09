import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import unittest
from unittest import mock


SCRIPT_DIR = os.path.dirname(__file__)
for env_var in ("GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    os.environ.setdefault(env_var, "unit-test")

from generate_summary import build_rows
from detect_log_failures import classify_log_file
from summarize_xml_testreports import _test_config_from_dir

loader = importlib.machinery.SourceFileLoader(
    "slow_parity_downloader", os.path.join(SCRIPT_DIR, "download_testlogs")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
downloader = importlib.util.module_from_spec(spec)
loader.exec_module(downloader)


class SlowParityProducerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(SCRIPT_DIR, "parity_job_config.json")) as config:
            cls.config = json.load(config)

    def test_slow_sources_are_registered_only_for_mi350(self):
        cuda = self.config["cuda"]
        self.assertEqual(
            cuda["slow"],
            [{
                "workflow": "slow",
                "job_prefix": "linux-jammy-cuda13.2-py3.11-gcc11-sm86",
            }],
        )
        self.assertEqual(cuda["shard_counts"]["slow"], 3)
        slow_regex = cuda["slow_checkrun_regex"]
        self.assertEqual(
            slow_regex,
            (
                r"^linux-jammy-cuda13[.]2-py3[.][0-9]+-gcc[0-9]+-sm86 "
                r"/ (test-osdc|test) [(]slow,"
            ),
        )
        self.assertRegex(
            "linux-jammy-cuda13.2-py3.11-gcc11-sm86 / test (slow, 1, 3, runner)",
            re.compile(slow_regex),
        )
        self.assertRegex(
            "linux-jammy-cuda13.2-py3.12-gcc14-sm86 / test-osdc (slow, 1, 3, runner)",
            re.compile(slow_regex),
        )
        self.assertNotRegex(
            "linux-jammy-cuda13.4-py3.11-gcc11-sm86 / test (slow, 1, 3, runner)",
            re.compile(slow_regex),
        )

        slow_rocm_arches = [
            arch for arch, config in self.config["rocm"].items()
            if "slow" in config
        ]
        self.assertEqual(slow_rocm_arches, ["mi350"])
        self.assertEqual(
            self.config["rocm"]["mi350"]["slow"],
            [{
                "workflow": "slow",
                "job_prefix": "linux-noble-rocm-py3.11-mi350",
            }],
        )
        self.assertEqual(
            self.config["rocm"]["mi350"]["shard_counts"]["slow"], 3
        )

    def test_xml_directory_classifies_slow(self):
        self.assertEqual(_test_config_from_dir("test-slow-2-3_123456"), "slow")

    def test_job_prefix_resolution_stays_in_accelerator_family(self):
        jobs = [
            {"name": "linux-jammy-rocm-py3.11-mi200 / test (default, 1, 10, runner)"},
            {"name": "linux-noble-rocm-py3.11-mi350 / test (default, 1, 8, runner)"},
        ]
        configured = "linux-jammy-rocm-py3.10-mi350"
        with mock.patch.object(downloader, "get_workflow_jobs", return_value=jobs):
            resolved = downloader.resolve_job_prefix({}, "default", configured)
        self.assertEqual(resolved, "linux-noble-rocm-py3.11-mi350")

        with mock.patch.object(
            downloader, "get_workflow_jobs", return_value=jobs[:1]
        ):
            resolved = downloader.resolve_job_prefix({}, "default", configured)
        self.assertEqual(resolved, configured)

    def test_downloader_matches_real_slow_jobs_and_scopes_arches(self):
        workflows, prefixes, _ = downloader.parity_config_views(
            self.config["rocm"]["mi350"]
        )
        self.assertEqual(workflows["slow"], "slow")
        self.assertEqual(
            prefixes["slow"], "linux-noble-rocm-py3.11-mi350"
        )
        mi300_workflows, _, _ = downloader.parity_config_views(
            self.config["rocm"]["mi300"]
        )
        self.assertNotIn("slow", mi300_workflows)

        jobs = [
            {"name": (
                "linux-noble-rocm-py3.11-mi350 / test "
                "(slow, 1, 3, linux.rocm.gpu.gfx950.1, module:rocm)"
            )},
            {"name": (
                "linux-jammy-cuda13.2-py3.11-gcc11-sm86 / test "
                "(slow, 1, 3, lf-l-x86aavx2-29-113-a10g)"
            )},
        ]
        _, rocm_jobs = downloader.get_test_jobs_for_config(
            jobs, "linux-noble-rocm-py3.11-mi350", "slow"
        )
        _, cuda_jobs = downloader.get_test_jobs_for_config(
            jobs, "linux-jammy-cuda13.2-py3.11-gcc11-sm86", "slow"
        )
        self.assertEqual(len(rocm_jobs), 1)
        self.assertEqual(len(cuda_jobs), 1)

    def test_log_failure_detector_classifies_slow(self):
        self.assertEqual(
            classify_log_file("rocm_slow2.txt"), ("rocm", "slow", 2)
        )
        self.assertEqual(
            classify_log_file("cuda_slow3.txt"), ("cuda", "slow", 3)
        )
        self.assertEqual(
            classify_log_file("eee2ee58_rocm_slow1.txt"),
            ("eee2ee58", "slow", 1),
        )

    def test_summary_has_slow_section_and_counts_it_in_overall(self):
        args = argparse.Namespace(
            sha="", pr_id="", set1_name="rocm", set2_name="cuda"
        )
        rows = [
            {
                "test_config": "slow",
                "status_rocm": "SKIPPED",
                "status_cuda": "PASSED",
                "running_time_rocm": "1.0",
                "running_time_cuda": "2.0",
            },
            {
                "test_config": "slow",
                "status_rocm": "SKIPPED",
                "status_cuda": "SKIPPED",
                "running_time_rocm": "0.0",
                "running_time_cuda": "0.0",
            },
        ]
        arch_data = {
            "mi350": {
                "rows": rows,
                "cols": (
                    "status_rocm", "status_cuda",
                    "running_time_rocm", "running_time_cuda",
                ),
                "has_set2": True,
            }
        }

        summary = build_rows(args, ["mi350"], arch_data)
        sections = [value for label, value in summary if label == "__section__"]
        self.assertIn("TEST SLOW", sections)
        overall_disagree = next(
            values for label, values in summary if label == "Overall DISAGREE%"
        )
        self.assertEqual(overall_disagree, ["100.00%"])
        total_cuda = next(
            values for label, values in summary if label == "TOTAL CUDA"
        )
        total_rocm = next(
            values for label, values in summary if label == "TOTAL ROCM"
        )
        self.assertEqual(total_cuda, [1])
        self.assertEqual(total_rocm, [1])

    def test_workflows_forward_and_arch_scope_slow(self):
        repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
        with open(os.path.join(repo_root, ".github/workflows/parity.yml")) as f:
            parity = f.read()
        with open(
            os.path.join(repo_root, ".github/workflows/parity-auto.yml")
        ) as f:
            parity_auto = f.read()
        with open(os.path.join(SCRIPT_DIR, "download_testlogs")) as f:
            downloader = f.read()

        self.assertIn("exclude_slow:", parity)
        self.assertIn('ARGS="$ARGS --exclude_slow"', parity)
        self.assertIn('if [ -n "$slow_archs" ]; then', parity_auto)
        self.assertIn('SLOW_EXCLUDE_FLAG="-f exclude_slow=true"', parity_auto)
        self.assertIn(".cuda.slow_checkrun_regex", parity_auto)
        self.assertIn("inputs.max_dispatches || '50'", parity_auto)
        self.assertIn("DISPATCH_STAGGER_SECONDS: '120'", parity_auto)
        self.assertIn(
            "some test shards are missing - skip", parity_auto
        )
        self.assertIn(
            "transient API failure; deferring this commit", parity_auto
        )
        self.assertIn("test-reports-test-slow-{i}-{slow_shards}", downloader)
        self.assertIn(
            'derive_shard_count(\n'
            '                slow_wf_rocm, slow_job_prefix, "slow"',
            downloader,
        )


if __name__ == "__main__":
    unittest.main()
