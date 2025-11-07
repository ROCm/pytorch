# mypy: allow-untyped-defs


import torch

from ..cpp_utils import DTYPE_TO_CPP


DTYPE_TO_ROCM_TYPE = {
    **DTYPE_TO_CPP,
    torch.float16: "uint16_t",
    torch.float8_e4m3fnuz: "uint8_t",
    torch.float8_e5m2fnuz: "uint8_t",
<<<<<<< HEAD
    torch.float8_e4m3fn: "uint8_t",
    torch.float8_e5m2: "uint8_t",
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    torch.bfloat16: "uint16_t",
}
