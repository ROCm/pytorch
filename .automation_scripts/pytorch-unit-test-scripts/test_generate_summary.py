import unittest

import generate_summary as gs


class HudAliasAndParityScopeTest(unittest.TestCase):
    def test_hud_display_label_uses_navi31_for_rx7900(self):
        config = {
            "rocm": {
                "rx7900": {"hud_aliases": ["navi31"], "parity_scope": "rocm_present"},
                "mi300": {},
            }
        }
        self.assertEqual(gs.hud_display_label("rx7900", config), "navi31")
        self.assertEqual(gs.hud_display_label("mi300", config), "mi300")

    def test_arch_parity_scope_defaults_to_full(self):
        config = {"rocm": {"mi300": {}, "rx7900": {"parity_scope": "rocm_present"}}}
        self.assertEqual(gs.arch_parity_scope("mi300", config), "full")
        self.assertEqual(gs.arch_parity_scope("rx7900", config), "rocm_present")

    def test_rocm_present_scope_excludes_missed_from_agree(self):
        rows = [
            {"test_config": "default", "status_rocm": "PASSED", "status_cuda": "PASSED"},
            {"test_config": "default", "status_rocm": "SKIPPED", "status_cuda": "PASSED"},
            {"test_config": "default", "status_rocm": "MISSED", "status_cuda": "PASSED"},
            {"test_config": "default", "status_rocm": "MISSED", "status_cuda": "PASSED"},
        ]
        full = gs.compute_test_config_stats(
            rows, "status_rocm", "status_cuda", "rocm", "cuda", parity_scope="full",
        )
        subset = gs.compute_test_config_stats(
            rows, "status_rocm", "status_cuda", "rocm", "cuda",
            parity_scope="rocm_present",
        )
        # Full scope: 1 skip + 2 miss against 4 CUDA = 75% disagree.
        self.assertEqual(full["DISAGREE [(SKIPPED+MISSED)/CUDA] %"], "75.00%")
        # Subset scope: only the 2 ROCm-present rows; 1 skip against 2 CUDA = 50%.
        self.assertEqual(subset["DISAGREE [(SKIPPED+MISSED)/CUDA] %"], "50.00%")
        self.assertEqual(subset["CUDA"], 2)
        self.assertEqual(subset["ROCM"], 2)
        self.assertEqual(
            subset["MISSED (on rocm, PASSED on cuda)"], 0,
        )

    def test_live_config_wires_rx7900_hud_alias_and_scope(self):
        config = gs.load_parity_config()
        self.assertEqual(gs.hud_display_label("rx7900", config), "navi31")
        self.assertEqual(gs.arch_parity_scope("rx7900", config), "rocm_present")


if __name__ == "__main__":
    unittest.main()
