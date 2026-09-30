import unittest

from auto_classify_skip_reasons import classify_test


class AutoClassifySkipReasonsTest(unittest.TestCase):
    def test_accelerator_device_requirement(self):
        for message in (
            "Need at least 4 CUDA devices",
            "Need at least 4 accelerator devices",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    classify_test(
                        message,
                        "distributed.tensor.test_common_rules",
                        "CommonRulesTest",
                        "test_pointwise_rules_broadcasting",
                    ),
                    "Greater than 4 GPU",
                )


if __name__ == "__main__":
    unittest.main()
