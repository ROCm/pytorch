import unittest

from job_name_match import choose_test_job_family


def job(name, job_id):
    return {"name": name, "id": job_id}


class ChooseTestJobFamilyTest(unittest.TestCase):
    def test_discovers_renamed_cuda_family(self):
        jobs = [
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test "
                f"(default, {shard}, 14, runner)",
                shard,
            )
            for shard in range(1, 15)
        ]

        family = choose_test_job_family(
            jobs,
            "default",
            "cuda",
            "linux-jammy-cuda13.0-py3.10-gcc11",
        )

        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.10-gcc11")
        self.assertEqual(family["total"], 14)

    def test_prefers_normal_cuda_family_over_debug(self):
        jobs = [
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test "
                f"(default, {shard}, 14, runner)",
                shard,
            )
            for shard in range(1, 15)
        ]
        jobs.extend(
            job(
                f"linux-jammy-cuda13.0-py3.10-gcc11-debug / test "
                f"(default, {shard}, 7, runner)",
                100 + shard,
            )
            for shard in range(1, 8)
        )

        family = choose_test_job_family(
            jobs,
            "default",
            "cuda",
            "linux-jammy-cuda13.0-py3.10-gcc11",
        )

        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.10-gcc11")

    def test_prefers_complete_renamed_family_over_incomplete_exact_match(self):
        jobs = [
            job(
                "linux-jammy-cuda13.0-py3.10-gcc11 / test "
                "(default, 1, 14, runner)",
                1,
            )
        ]
        jobs.extend(
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test "
                f"(default, {shard}, 14, runner)",
                100 + shard,
            )
            for shard in range(1, 15)
        )

        family = choose_test_job_family(
            jobs,
            "default",
            "cuda",
            "linux-jammy-cuda13.0-py3.10-gcc11",
        )

        self.assertEqual(family["prefix"], "linux-jammy-cuda13.2-py3.10-gcc11")

    def test_discovers_historical_rocm_family(self):
        jobs = [
            job(
                f"linux-jammy-rocm-py3.10-mi350 / test "
                f"(distributed, {shard}, 3, runner)",
                shard,
            )
            for shard in range(1, 4)
        ]

        family = choose_test_job_family(
            jobs,
            "distributed",
            "rocm",
            "linux-noble-rocm-py3.11-mi350",
        )

        self.assertEqual(family["prefix"], "linux-jammy-rocm-py3.10-mi350")
        self.assertEqual(family["total"], 3)

    def test_preserves_discovered_test_kind(self):
        jobs = [
            job(
                f"linux-jammy-cuda13.2-py3.10-gcc11 / test-osdc "
                f"(distributed, {shard}, 10, runner)",
                shard,
            )
            for shard in range(1, 11)
        ]

        family = choose_test_job_family(
            jobs,
            "distributed",
            "cuda",
            "linux-jammy-cuda13.2-py3.10-gcc11",
        )

        self.assertEqual(family["kind"], "test-osdc")

    def test_marks_missing_shards_incomplete(self):
        jobs = [
            job(
                "linux-jammy-cuda13.2-py3.10-gcc11 / test "
                "(default, 1, 14, runner)",
                1,
            )
        ]

        family = choose_test_job_family(
            jobs,
            "default",
            "cuda",
            "linux-jammy-cuda13.2-py3.10-gcc11",
        )

        self.assertFalse(family["complete"])

    def test_keeps_platforms_separate(self):
        jobs = [
            job(
                "linux-jammy-cuda13.2-py3.10-gcc11 / test "
                "(default, 1, 14, runner)",
                1,
            )
        ]

        self.assertIsNone(
            choose_test_job_family(
                jobs,
                "default",
                "rocm",
                "linux-noble-rocm-py3.11-mi350",
            )
        )


if __name__ == "__main__":
    unittest.main()
