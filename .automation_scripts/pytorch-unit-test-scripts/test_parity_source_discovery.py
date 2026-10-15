import json
import os
import unittest

from parity_source_discovery import AmbiguousSourceError, resolve_topology


HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "parity_job_config.json")) as config_file:
    CONFIG = json.load(config_file)


def run(run_id, path, event="push", created_at="2026-10-07T00:00:00Z"):
    return {
        "id": run_id,
        "path": path,
        "event": event,
        "created_at": created_at,
    }


def family(run_id, prefix, config, total, first_id, conclusion="success"):
    return [
        {
            "id": first_id + shard,
            "name": (
                f"{prefix} / test ({config}, {shard}, {total}, "
                "mt-l-x86aavx2-11-41-l4)"
            ),
            "status": "completed",
            "conclusion": conclusion,
            "details_url": (
                f"https://github.com/pytorch/pytorch/actions/runs/{run_id}/"
                f"job/{first_id + shard}"
            ),
        }
        for shard in range(1, total + 1)
    ]


class ParitySourceDiscoveryTest(unittest.TestCase):
    def test_transition_sha_prefers_complete_trunk_bundle(self):
        checks = []
        for config, total, first_id in (
            ("default", 14, 100),
            ("distributed", 10, 200),
            ("inductor", 2, 300),
        ):
            checks += family(
                10, "linux-jammy-cuda13.2-py3.11-gcc11",
                config, total, first_id,
            )
        checks += family(
            20, "unit-test / inductor-test-cuda132", "inductor", 2, 400
        )

        topology = resolve_topology(
            checks,
            [
                run(10, ".github/workflows/trunk.yml"),
                run(20, ".github/workflows/inductor-unittest.yml"),
            ],
            CONFIG,
            archs=[],
        )

        self.assertEqual(topology["cuda"]["inductor"]["run_id"], 10)
        self.assertEqual(topology["cuda"]["inductor"]["total"], 2)

    def test_historical_sha_uses_configured_fallback_hint(self):
        checks = family(
            20, "unit-test / inductor-test", "inductor", 2, 100
        )
        checks += family(
            20, "unit-test / inductor-test-cuda132", "inductor", 2, 200
        )
        topology = resolve_topology(
            checks,
            [run(20, ".github/workflows/inductor.yml")],
            CONFIG,
            archs=[],
        )

        self.assertEqual(
            topology["cuda"]["inductor"]["prefix"],
            "unit-test / inductor-test-cuda132",
        )

    def test_workflow_and_prefix_rename_are_discovered(self):
        checks = family(
            30, "linux-future-cuda14-py3.13-clang", "default", 3, 100
        )
        topology = resolve_topology(
            checks,
            [run(30, ".github/workflows/future-ci.yml")],
            CONFIG,
            archs=[],
        )

        self.assertEqual(topology["cuda"]["default"]["run_id"], 30)
        self.assertEqual(topology["cuda"]["default"]["total"], 3)

    def test_incomplete_family_is_reported_but_not_selected_as_complete(self):
        checks = family(
            30, "linux-future-cuda14-py3.13-clang", "default", 3, 100
        )[:-1]
        topology = resolve_topology(
            checks,
            [run(30, ".github/workflows/future-ci.yml")],
            CONFIG,
            archs=[],
        )

        self.assertFalse(topology["cuda"]["default"]["complete"])

    def test_preview_wins_over_mi350_runner_alias(self):
        checks = family(
            40,
            "linux-noble-rocm-preview-py3.12-mi350",
            "default",
            2,
            100,
        )
        topology = resolve_topology(
            checks,
            [run(40, ".github/workflows/renamed-preview.yml", event="schedule")],
            CONFIG,
            archs=["preview", "mi350"],
        )

        self.assertEqual(topology["rocm"]["preview"]["default"]["run_id"], 40)
        self.assertNotIn("default", topology["rocm"]["mi350"])

    def test_equally_suitable_different_workflows_fail_closed(self):
        checks = family(
            50, "linux-future-cuda14-py3.13-clang", "default", 2, 100
        )
        checks += family(
            60, "linux-future-cuda14-py3.13-clang", "default", 2, 200
        )

        with self.assertRaisesRegex(AmbiguousSourceError, "equally suitable"):
            resolve_topology(
                checks,
                [
                    run(50, ".github/workflows/one.yml"),
                    run(60, ".github/workflows/two.yml"),
                ],
                CONFIG,
                archs=[],
            )

    def test_variant_jobs_are_ignored(self):
        checks = family(
            70,
            "linux-jammy-cuda13.2-py3.11-gcc11-rerun_disabled_tests",
            "default",
            2,
            100,
        )
        topology = resolve_topology(
            checks,
            [run(70, ".github/workflows/trunk.yml", event="schedule")],
            CONFIG,
            archs=[],
        )

        self.assertIn("cuda/default", topology["missing"])

    def test_rerun_keeps_successful_and_cancelled_job_ids(self):
        checks = family(
            80, "linux-jammy-cuda13.2-py3.11-gcc11", "default", 2, 100
        )
        duplicate = dict(checks[0])
        duplicate.update({"id": 999, "conclusion": "cancelled"})
        checks.append(duplicate)

        topology = resolve_topology(
            checks,
            [run(80, ".github/workflows/trunk.yml")],
            CONFIG,
            archs=[],
        )

        selected = topology["cuda"]["default"]
        self.assertTrue(selected["complete"])
        self.assertEqual(selected["job_ids"], [101, 102, 999])

    def test_configured_path_separates_sibling_workflow(self):
        prefix = "linux-jammy-cuda13.2-py3.11-gcc11"
        checks = family(90, prefix, "default", 2, 100)
        checks += family(91, prefix, "default", 2, 200)

        topology = resolve_topology(
            checks,
            [
                run(90, ".github/workflows/trunk.yml"),
                run(91, ".github/workflows/trunk-rocm-sandbox.yml"),
            ],
            CONFIG,
            archs=[],
        )

        self.assertEqual(topology["cuda"]["default"]["run_id"], 90)


if __name__ == "__main__":
    unittest.main()
