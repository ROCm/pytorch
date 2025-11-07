# mypy: allow-untyped-defs
import torch

class MyAutogradFunction(torch.autograd.Function):
    @staticmethod
<<<<<<< HEAD
    # pyrefly: ignore [bad-override]
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def forward(ctx, x):
        return x.clone()

    @staticmethod
<<<<<<< HEAD
    # pyrefly: ignore [bad-override]
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def backward(ctx, grad_output):
        return grad_output + 1

class AutogradFunction(torch.nn.Module):
    """
    TorchDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """

    def forward(self, x):
        return MyAutogradFunction.apply(x)

example_args = (torch.randn(3, 2),)
model = AutogradFunction()
