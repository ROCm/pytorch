# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=legacy")
<<<<<<< HEAD
from torch.testing._internal.common_utils import parse_cmd_line_args, run_tests

if __name__ == '__main__':
    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from test_jit import *  # noqa: F403, F401
=======
from test_jit import *  # noqa: F403
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

if __name__ == '__main__':
    run_tests()
