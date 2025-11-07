from typing import Any

import torch.library
from torch import Tensor
from torch.autograd import Function


if not torch._running_with_deploy():
    _test_lib_def = torch.library.Library("_inductor_test", "DEF")
    _test_lib_def.define(
        "realize(Tensor self) -> Tensor", tags=torch.Tag.pt2_compliant_tag
    )

    _test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
<<<<<<< HEAD
    for dispatch_key in ("CPU", "CUDA", "Meta"):
=======
    for dispatch_key in ("CPU", "CUDA", "MPS", "Meta"):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)

    class Realize(Function):
        @staticmethod
        def forward(ctx: object, x: Tensor) -> Tensor:
            return torch.ops._inductor_test.realize(x)

        @staticmethod
        # types need to stay consistent with _SingleLevelFunction
        def backward(ctx: Any, *grad_output: Any) -> Any:
            return grad_output[0]

    def realize(x: Tensor) -> Tensor:
        return Realize.apply(x)
