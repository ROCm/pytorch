# flake8: noqa
from typing import Any
from typing_extensions import assert_type

from torch import randn, Tensor


# See ../pass/arithmetic_ops.py for more information

TENSOR, INT, FLOAT = randn(3), 2, 1.5

<<<<<<< HEAD
assert_type(
    INT & TENSOR,  # E: Unsupported operand types for & ("int" and "Tensor")  [operator]
    Any,
)
assert_type(
    INT | TENSOR,  # E: Unsupported operand types for | ("int" and "Tensor")  [operator]
    Any,
)
assert_type(
    INT ^ TENSOR,  # E: Unsupported operand types for ^ ("int" and "Tensor")  [operator]
    Any,
)

assert_type(
    FLOAT  # E: Unsupported operand types for & ("float" and "Tensor")  [operator]
    & TENSOR,
    Tensor,
)
assert_type(
    FLOAT  # E: Unsupported operand types for | ("float" and "Tensor")  [operator]
    | TENSOR,
    Tensor,
)
assert_type(
    FLOAT  # E: Unsupported operand types for ^ ("float" and "Tensor")  [operator]
    ^ TENSOR,
    Tensor,
)
=======
FLOAT & TENSOR  # E: Unsupported operand types for & ("float" and "Tensor")
FLOAT | TENSOR  # E: Unsupported operand types for | ("float" and "Tensor")
FLOAT ^ TENSOR  # E: Unsupported operand types for ^ ("float" and "Tensor")
# FIXME: false negatives (https://github.com/pytorch/pytorch/issues/155701)
# TENSOR & FLOAT  # E: Unsupported operand types for & ("Tensor" and "float" )
# TENSOR | FLOAT  # E: Unsupported operand types for | ("Tensor" and "float" )
# TENSOR ^ FLOAT  # E: Unsupported operand types for ^ ("Tensor" and "float" )
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
