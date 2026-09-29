import unittest
from datetime import datetime, timezone

import dispatch_therock_ci as dispatch


class FakeClient:
    def __init__(self, runs=None):
        self.runs = runs or []
        self.requests = []

    def workflow_runs(self, repository, workflow_file):
        self.requests.append(("runs", repository, workflow_file))
        return self.runs


class RecordingClient(dispatch.GitHubClient):
    def __init__(self):
        super().__init__("token")
        self.requests = []

    def request(self, method, path, payload=None):
        self.requests.append((method, path, payload))
        return None


class DispatchTheRockCITest(unittest.TestCase):
    def test_build_workflow_inputs(self):
        workflow = dispatch.WORKFLOWS[0]

        inputs = dispatch.build_workflow_inputs(
            workflow,
            sha="abc123",
            rocm_version="10.2.0a20260929",
            package_index_url="https://example.com/whl-next/",
        )

        self.assertEqual(
            inputs,
            {
                "artifact_group": "gfx94X-dcgpu",
                "python_version": "3.12",
                "pytorch_git_ref": "abc123",
                "rocm_package_index_url": "https://example.com/whl-next/",
                "rocm_version": "10.2.0a20260929",
                "cache_type": "sccache",
                "build_runs_on": "aws-linux-scale-rocm-prod",
            },
        )

    def test_select_dispatched_run_uses_sha_and_time(self):
        dispatched_at = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        runs = [
            {
                "id": 1,
                "display_title": "Build pyabc123",
                "created_at": "2026-09-29T11:59:59Z",
            },
            {
                "id": 2,
                "display_title": "Build pyother",
                "created_at": "2026-09-29T12:00:01Z",
            },
            {
                "id": 3,
                "display_title": "Build pyabc123",
                "created_at": "2026-09-29T12:00:02Z",
            },
            {
                "id": 4,
                "display_title": "Build pyabc123 retry",
                "created_at": "2026-09-29T12:00:03Z",
            },
        ]

        selected = dispatch.select_dispatched_run(
            runs, sha="abc123", dispatched_at=dispatched_at
        )

        self.assertEqual(selected["id"], 3)

    def test_status_state(self):
        self.assertEqual(dispatch.status_state("success"), "success")
        self.assertEqual(dispatch.status_state("failure"), "failure")
        self.assertEqual(dispatch.status_state("cancelled"), "error")
        self.assertEqual(dispatch.status_state("skipped"), "error")

    def test_all_success_requires_both_workflows(self):
        self.assertTrue(
            dispatch.all_success({"Linux": "success", "Windows": "success"})
        )
        self.assertFalse(
            dispatch.all_success({"Linux": "success", "Windows": "failure"})
        )
        self.assertFalse(dispatch.all_success({"Linux": "success"}))

    def test_wait_for_run_times_out(self):
        client = FakeClient()
        clock_values = iter([0.0, 0.1, 1.0])

        with self.assertRaisesRegex(TimeoutError, "Linux"):
            dispatch.wait_for_run(
                client,
                "ROCm/TheRock",
                dispatch.WORKFLOWS[0],
                sha="abc123",
                dispatched_at=datetime.now(timezone.utc),
                timeout_seconds=1,
                poll_interval=0,
                monotonic=lambda: next(clock_values),
                sleep=lambda _: None,
            )

    def test_cancel_run_calls_actions_api(self):
        client = RecordingClient()

        client.cancel_run("ROCm/TheRock", 12345)

        self.assertEqual(
            client.requests,
            [
                (
                    "POST",
                    "/repos/ROCm/TheRock/actions/runs/12345/cancel",
                    None,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
