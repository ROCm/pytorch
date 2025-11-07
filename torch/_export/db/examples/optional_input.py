# mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class OptionalInput(torch.nn.Module):
    """
    Tracing through optional input is not supported yet
    """

    def forward(self, x, y=torch.randn(2, 3)):
        if y is not None:
            return x + y
        return x


example_args = (torch.randn(2, 3),)
tags = {"python.object-model"}
<<<<<<< HEAD
support_level = SupportLevel.SUPPORTED
=======
support_level = SupportLevel.NOT_SUPPORTED_YET
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
model = OptionalInput()
