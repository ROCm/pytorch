import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


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


if __name__ == "__main__":
    unittest.main()
