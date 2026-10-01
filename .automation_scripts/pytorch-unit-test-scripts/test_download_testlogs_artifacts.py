import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from detect_log_failures import classify_log_file


def _load_download_testlogs():
    for var in ("GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        os.environ.setdefault(var, "test")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_testlogs")
    loader = importlib.machinery.SourceFileLoader("download_testlogs", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


dtl = _load_download_testlogs()

THROTTLE_ERROR_DOC = (
    b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?><Error><Code>ServerBusy'
    b"</Code><Message>Egress is over the account limit.\n</Message></Error>"
)


class Response:
    headers = {}

    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class CorruptArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        dtl.error_msgs.clear()
        self.addCleanup(dtl.error_msgs.clear)

    def _stub_downloads(self, names):
        for name, payload in names.items():
            if payload is None:
                with zipfile.ZipFile(name, "w") as archive:
                    archive.writestr("test-reports/TEST-good.xml", "<testsuite/>")
            else:
                Path(name).write_bytes(payload)

        def fake_s3(prefix, run_id, attempt, allowed_substrings=None):
            return [Path(name) for name in names if name.startswith(prefix)]

        patcher = mock.patch.object(dtl, "download_s3_artifacts", fake_s3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_non_zip_artifact_is_skipped_not_fatal(self):
        self._stub_downloads(
            {
                "test-reports-test-default-1-2-rocm.gpu_1.zip": None,
                "test-reports-test-default-2-2-rocm.gpu_2.zip": THROTTLE_ERROR_DOC,
            }
        )

        dtl.download_xml_files(123, 1, prefixes=["test-reports-test-default"])

        self.assertEqual(len(list(Path(".").glob("**/TEST-good.xml"))), 1)
        joined = "\n".join(dtl.error_msgs)
        self.assertIn("test-reports-test-default-2-2-rocm.gpu_2.zip", joined)
        self.assertIn("is not a zip file", joined)
        self.assertIn("Egress is over the account limit", joined)

    def test_all_artifacts_corrupt_records_every_shard(self):
        self._stub_downloads(
            {
                "test-reports-test-default-1-2-rocm.gpu_1.zip": THROTTLE_ERROR_DOC,
                "test-reports-test-default-2-2-rocm.gpu_2.zip": THROTTLE_ERROR_DOC,
            }
        )

        dtl.download_xml_files(123, 1, prefixes=["test-reports-test-default"])

        self.assertEqual(len(dtl.error_msgs), 2)

    def test_github_fallback_fills_each_missing_s3_prefix(self):
        s3_path = Path("test-reports-test-default-1-2-rocm.gpu_1.zip")
        gha_path = Path("test-reports-test-default-2-2-rocm.gpu_2.zip")
        for path in (s3_path, gha_path):
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    f"test-reports/{path.stem}.xml", "<testsuite/>"
                )

        def fake_s3(prefix, run_id, attempt, allowed_substrings=None):
            return [s3_path] if prefix.endswith("1-2") else []

        with (
            mock.patch.object(dtl, "download_s3_artifacts", side_effect=fake_s3),
            mock.patch.object(
                dtl,
                "download_gha_artifacts_filtered",
                return_value=[gha_path],
            ) as gha,
        ):
            dtl.download_xml_files(
                123,
                1,
                prefixes=[
                    "test-reports-test-default-1-2",
                    "test-reports-test-default-2-2",
                ],
            )

        self.assertEqual(len(list(Path(".").glob("**/*.xml"))), 2)
        self.assertEqual(
            gha.call_args.kwargs["prefixes"],
            ["test-reports-test-default-2-2"],
        )


class ApiRetryTest(unittest.TestCase):
    def setUp(self):
        dtl.authentication_headers = {}

    def test_jobs_retry_connection_drop_and_secondary_limit(self):
        responses = [
            dtl.requests.exceptions.ConnectionError(),
            Response(403, {}),
            Response(200, {"jobs": [], "total_count": 0}),
        ]
        with (
            mock.patch.object(dtl.requests, "get", side_effect=responses) as get,
            mock.patch.object(dtl.time, "sleep") as sleep,
        ):
            result = dtl._fetch_jobs_page("jobs", {}, max_retries=3)

        self.assertEqual(result["jobs"], [])
        self.assertEqual(get.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_workflow_runs_retry_bad_response(self):
        responses = [
            Response(503, {}),
            Response(200, {"workflow_runs": [{"id": 1}]}),
        ]
        with (
            mock.patch.object(dtl.requests, "get", side_effect=responses),
            mock.patch.object(dtl.time, "sleep") as sleep,
        ):
            result = dtl._fetch_workflow_runs("trunk", {}, max_retries=2)

        self.assertEqual(result, [{"id": 1}])
        sleep.assert_called_once_with(1)


class CrossSourceComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.tmp.name)
        dtl.error_msgs.clear()
        self.addCleanup(dtl.error_msgs.clear)

    def _args(self, **overrides):
        values = {
            "set1_source": "trunk",
            "set1_sha": "a" * 40,
            "set2_source": "preview",
            "set2_sha": "b" * 40,
            "baseline_sha": None,
            "pr_id": None,
            "ignore_status": True,
            "artifacts_only": True,
            "include_inductor_periodic": False,
            "created": None,
            "max_pages": 10,
            "exclude_default": False,
            "exclude_distributed": True,
            "exclude_inductor": True,
            "exclude_slow": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_side_commands_select_platform_and_arch(self):
        args = self._args()
        trunk = dtl._cross_side_command("trunk", "a" * 40, args)
        preview = dtl._cross_side_command("preview", "b" * 40, args)

        self.assertIn("--no_rocm", trunk)
        self.assertEqual(trunk[trunk.index("--arch") + 1], "mi350")
        self.assertIn("--no_cuda", preview)
        self.assertEqual(preview[preview.index("--arch") + 1], "preview")

        periodic = dtl._cross_side_command(
            "trunk", "a" * 40,
            self._args(include_inductor_periodic=True),
        )
        self.assertIn("--include_inductor_periodic", periodic)

    def test_comparison_downloads_each_side_to_separate_directories(self):
        args = self._args()
        commands = []

        def fake_run(command, check, cwd):
            commands.append(command)
            sha = command[command.index("--sha1") + 1]
            xml_name = "cuda_xml" if "--no_rocm" in command else "rocm_xml"
            source = command[command.index("--arch") + 1]
            folder = Path(cwd) / f"20260930_{sha}"
            xml_dir = folder / xml_name / "shard"
            xml_dir.mkdir(parents=True, exist_ok=True)
            (xml_dir / "TEST-result.xml").write_text(
                f'<testsuite name="{source}"/>'
            )
            (xml_dir / "_wf_run_ids.json").write_text('{"1": "123"}')
            (folder / f"{xml_name[:-4]}1.txt").write_text("test log")
            return SimpleNamespace(returncode=0)

        with mock.patch.object(dtl.subprocess, "run", side_effect=fake_run):
            folder = dtl.run_cross_source_comparison(args)

        self.assertTrue((folder / "set1_xml/shard/TEST-result.xml").is_file())
        self.assertTrue((folder / "set2_xml/shard/TEST-result.xml").is_file())
        self.assertFalse((folder / "comparison_sources.json").exists())
        self.assertTrue(all("--exclude_slow" in command for command in commands))
        self.assertTrue((folder / f"trunk@{'a' * 8}_cuda1.txt").is_file())
        self.assertTrue((folder / f"preview@{'b' * 8}_rocm1.txt").is_file())

    def test_same_sha_rocm_sources_remain_isolated(self):
        sha = "c" * 40
        args = self._args(
            set1_source="mi350", set1_sha=sha,
            set2_source="preview", set2_sha=sha,
        )

        def fake_run(command, check, cwd):
            source = command[command.index("--arch") + 1]
            xml_dir = Path(cwd) / f"20260930_{sha}" / "rocm_xml"
            xml_dir.mkdir(parents=True, exist_ok=True)
            (xml_dir / "TEST-result.xml").write_text(source)
            return SimpleNamespace(returncode=0)

        with mock.patch.object(dtl.subprocess, "run", side_effect=fake_run):
            folder = dtl.run_cross_source_comparison(args)

        self.assertEqual((folder / "set1_xml/TEST-result.xml").read_text(), "mi350")
        self.assertEqual((folder / "set2_xml/TEST-result.xml").read_text(), "preview")

    def test_comparison_requires_all_four_source_fields(self):
        with self.assertRaisesRegex(ValueError, "requires set1_source"):
            dtl.run_cross_source_comparison(self._args(set2_sha=None))

    def test_missing_side_xml_is_recorded_as_failure(self):
        with mock.patch.object(
            dtl.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ):
            dtl.run_cross_source_comparison(self._args())

        self.assertIn(
            f"No trunk XML reports found for {'a' * 40}",
            dtl.error_msgs,
        )
        self.assertIn(
            f"No preview XML reports found for {'b' * 40}",
            dtl.error_msgs,
        )

    def test_legacy_args_do_not_enable_cross_source_mode(self):
        with mock.patch.object(sys, "argv", ["download_testlogs"]):
            args = dtl.parse_args()

        self.assertIsNone(args.set1_source)
        self.assertIsNone(args.set2_source)
        self.assertIsNone(args.set1_sha)
        self.assertIsNone(args.set2_sha)

    def test_cross_source_log_label_is_preserved(self):
        self.assertEqual(
            classify_log_file("trunk@1234abcd_cuda_dist2.txt"),
            ("trunk@1234abcd", "distributed", 2),
        )


if __name__ == "__main__":
    unittest.main()
