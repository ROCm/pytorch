# Owner(s): ["oncall: jit"]

import torch._lazy.metrics
<<<<<<< HEAD
=======
from torch.testing._internal.common_utils import run_tests
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


def test_metrics():
    names = torch._lazy.metrics.counter_names()
    assert len(names) == 0, f"Expected no counter names, but got {names}"
<<<<<<< HEAD
=======


if __name__ == "__main__":
    run_tests()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
