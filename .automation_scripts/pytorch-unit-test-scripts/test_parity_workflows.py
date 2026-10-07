import os
import unittest

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def workflow(name):
    with open(os.path.join(ROOT, ".github", "workflows", name)) as file:
        return yaml.safe_load(file)


class RuntimeDiscoveryWorkflowTest(unittest.TestCase):
    def test_auto_trigger_uses_runtime_resolver_and_passes_manifest(self):
        data = workflow("parity-auto.yml")
        steps = data["jobs"]["scan-and-dispatch"]["steps"]
        script = steps[-1]["run"]

        self.assertEqual(steps[0]["name"], "Checkout")
        self.assertIn("resolve_parity_sources.py", script)
        self.assertIn('--sha "$sha"', script)
        self.assertIn("repos/$UPSTREAM/commits?", script)
        self.assertIn("-f source_manifest_b64=", script)
        self.assertNotIn('archs_that_ran "$sha"', script)

    def test_auto_trigger_fails_after_sustained_missing_cuda_role(self):
        script = workflow("parity-auto.yml")[
            "jobs"
        ]["scan-and-dispatch"]["steps"][-1]["run"]

        self.assertIn("CUDA_MISSING_INDUCTOR=0", script)
        self.assertIn('if [ "$streak" -ge 3 ]', script)
        self.assertIn("runtime topology drift", script)
        self.assertIn("2h settlement window", script)

    def test_parity_forwards_manifest_to_downloader(self):
        data = workflow("parity.yml")
        inputs = data[True]["workflow_dispatch"]["inputs"]
        step = next(
            step
            for step in data["jobs"]["generate-parity"]["steps"]
            if step.get("name") == "Download artifacts"
        )

        self.assertIn("source_manifest_b64", inputs)
        self.assertIn("--source_manifest_b64", step["run"])


if __name__ == "__main__":
    unittest.main()
