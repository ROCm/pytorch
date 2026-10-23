import importlib.machinery
import importlib.util
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(__file__))

# download_testlogs is a CLI without a .py suffix and refuses to import without
# credentials, so load it by path with placeholders.
for _var in ("GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_var, "placeholder")

_PATH = os.path.join(os.path.dirname(__file__), "download_testlogs")
_spec = importlib.util.spec_from_loader(
    "download_testlogs", importlib.machinery.SourceFileLoader("download_testlogs", _PATH)
)
dtl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dtl)


# slow.yml runs for fc6c9cdfd4c55eb9a9df3f3923b8d808c1cf1ae3: the push run holds
# the real shards, the scheduled run re-ran only the rerun_disabled_tests
# variants a minute later and so comes back first from the API.
PUSH_RUN = 35976419544
VARIANT_RUN = 35976562424
ROCM_PREFIX = "linux-noble-rocm-py3.11-mi350"
CUDA_PREFIX = "linux-jammy-cuda13.0-py3.10-gcc11-sm86"


def _jobs(run_id, prefix):
    suffix = ", rerun_disabled_tests" if run_id == VARIANT_RUN else ""
    base = 10756412000 if run_id == VARIANT_RUN else 10756383000
    return [
        {"id": base + i, "name": f"{prefix} / test (slow, {i}, 3, runner{suffix})"}
        for i in (1, 2, 3)
    ]


class SlowRunSelectionTest(unittest.TestCase):
    def setUp(self):
        self.runs = [
            {"id": VARIANT_RUN, "event": "schedule", "status": "completed"},
            {"id": PUSH_RUN, "event": "push", "status": "completed"},
        ]
        self.prefix = ROCM_PREFIX
        self._orig = (dtl.requests.get, dtl.get_workflow_jobs)

        class Resp:
            def json(_self):
                return {"workflow_runs": self.runs}

        dtl.requests.get = lambda *a, **k: Resp()
        dtl.get_workflow_jobs = lambda run, all_attempts=False: _jobs(run["id"], self.prefix)

    def tearDown(self):
        dtl.requests.get, dtl.get_workflow_jobs = self._orig

    def test_picks_push_run_over_variant_rerun(self):
        for prefix in (ROCM_PREFIX, CUDA_PREFIX):
            self.prefix = prefix
            run = dtl.resolve_non_variant_run("slow", "fc6c9cd", "slow", prefix)
            self.assertEqual(run["id"], PUSH_RUN)

    def test_declines_when_only_variant_shards_exist(self):
        self.runs = [{"id": VARIANT_RUN, "event": "schedule", "status": "completed"}]
        self.assertIsNone(
            dtl.resolve_non_variant_run("slow", "fc6c9cd", "slow", ROCM_PREFIX)
        )


if __name__ == "__main__":
    unittest.main()
