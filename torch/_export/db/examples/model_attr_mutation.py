# mypy: allow-untyped-defs
import torch
<<<<<<< HEAD
=======
from torch._export.db.case import SupportLevel
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class ModelAttrMutation(torch.nn.Module):
    """
<<<<<<< HEAD
    Attribute mutation raises a warning. Covered in the test_export.py test_detect_leak_strict test.
=======
    Attribute mutation is not supported.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """

    def __init__(self) -> None:
        super().__init__()
        self.attr_list = [torch.randn(3, 2), torch.randn(3, 2)]

    def recreate_list(self):
        return [torch.zeros(3, 2), torch.zeros(3, 2)]

    def forward(self, x):
        self.attr_list = self.recreate_list()
        return x.sum() + self.attr_list[0].sum()


example_args = (torch.randn(3, 2),)
tags = {"python.object-model"}
<<<<<<< HEAD
=======
support_level = SupportLevel.NOT_SUPPORTED_YET
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
model = ModelAttrMutation()
