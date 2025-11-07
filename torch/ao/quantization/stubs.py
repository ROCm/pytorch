<<<<<<< HEAD
# mypy: allow-untyped-defs

from torch import nn
=======
from typing import Any, Optional

import torch
from torch import nn
from torch.ao.quantization import QConfig


__all__ = ["QuantStub", "DeQuantStub", "QuantWrapper"]
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

<<<<<<< HEAD
    def __init__(self, qconfig=None):
=======
    def __init__(self, qconfig: Optional[QConfig] = None):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

<<<<<<< HEAD
    def forward(self, x):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return x


class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

<<<<<<< HEAD
    def __init__(self, qconfig=None):
=======
    def __init__(self, qconfig: Optional[Any] = None):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

<<<<<<< HEAD
    def forward(self, x):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return x


class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
<<<<<<< HEAD
=======

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

<<<<<<< HEAD
    def __init__(self, module):
=======
    def __init__(self, module: nn.Module):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        super().__init__()
        qconfig = getattr(module, "qconfig", None)
        self.add_module("quant", QuantStub(qconfig))
        self.add_module("dequant", DeQuantStub(qconfig))
        self.add_module("module", module)
        self.train(module.training)

<<<<<<< HEAD
    def forward(self, X):
=======
    def forward(self, X: torch.Tensor) -> torch.Tensor:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)
