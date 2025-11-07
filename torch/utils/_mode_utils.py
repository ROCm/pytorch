# mypy: allow-untyped-defs
<<<<<<< HEAD
import torch
from typing import TypeVar

T = TypeVar('T')
=======
from typing import TypeVar

import torch


T = TypeVar("T")

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

# returns if all are the same mode
def all_same_mode(modes):
    return all(tuple(mode == modes[0] for mode in modes))

<<<<<<< HEAD
=======

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
no_dispatch = torch._C._DisableTorchDispatch
