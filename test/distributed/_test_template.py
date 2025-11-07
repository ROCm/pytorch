# Owner(s): ["oncall: distributed"]

<<<<<<< HEAD
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import run_tests


class TestTemplate(MultiProcContinuousTest):
=======
from torch.testing._internal.common_distributed import MultiProcContinousTest
from torch.testing._internal.common_utils import run_tests


class TestTemplate(MultiProcContinousTest):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def testABC(self):
        print(f"rank {self.rank} of {self.world_size} testing ABC")

    def testDEF(self):
        print(f"rank {self.rank} of {self.world_size} testing DEF")


if __name__ == "__main__":
    run_tests()
