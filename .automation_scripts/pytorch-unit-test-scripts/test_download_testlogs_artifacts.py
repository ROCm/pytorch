import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


def _load_download_testlogs():
    # download_testlogs has no .py suffix and checks these at import time.
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

# What upstream's artifact copy actually stored under a .zip key on
# 2026-09-28 when its storage backend throttled the transfer.
THROTTLE_ERROR_DOC = (
    b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?><Error><Code>ServerBusy'
    b"</Code><Message>Egress is over the account limit.\n</Message></Error>"
)


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
                with zipfile.ZipFile(name, "w") as zf:
                    zf.writestr("test-reports/TEST-good.xml", "<testsuite/>")
            else:
                Path(name).write_bytes(payload)

        def fake_s3(prefix, run_id, attempt, allowed_substrings=None):
            return [Path(n) for n in names if n.startswith(prefix)]

        original = dtl.download_s3_artifacts
        dtl.download_s3_artifacts = fake_s3
        self.addCleanup(setattr, dtl, "download_s3_artifacts", original)

    def test_non_zip_artifact_is_skipped_not_fatal(self):
        self._stub_downloads(
            {
                "test-reports-test-default-1-2-rocm.gpu_1.zip": None,
                "test-reports-test-default-2-2-rocm.gpu_2.zip": THROTTLE_ERROR_DOC,
            }
        )

        dtl.download_xml_files(123, 1, prefixes=["test-reports-test-default"])

        extracted = list(Path(".").glob("**/TEST-good.xml"))
        self.assertEqual(len(extracted), 1, "healthy shard must still be extracted")

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


if __name__ == "__main__":
    unittest.main()
