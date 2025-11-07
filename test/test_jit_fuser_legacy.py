# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=legacy")
<<<<<<< HEAD

if __name__ == "__main__":
    from torch.testing._internal.common_utils import parse_cmd_line_args

    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from test_jit_fuser import *  # noqa: F403

if __name__ == '__main__':
    run_tests()
