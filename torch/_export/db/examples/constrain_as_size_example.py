# mypy: allow-untyped-defs
import torch


class ConstrainAsSizeExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
<<<<<<< HEAD
    can trace further. Please look at torch._check APIs.
=======
    can trace further. Please look at torch._check and torch._check_is_size APIs.
    torch._check_is_size is used for values that NEED to be used for constructing
    tensor.
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """

    def forward(self, x):
        a = x.item()
<<<<<<< HEAD
        torch._check(a >= 0)
=======
        torch._check_is_size(a)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        torch._check(a <= 5)
        return torch.zeros((a, 5))


example_args = (torch.tensor(4),)
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsSizeExample()
