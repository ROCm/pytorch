# Owner(s): ["module: onnx"]

import torch


<<<<<<< HEAD
# Autograd function that is a replica of the autograd function in
=======
# Autograd funtion that is a replica of the autograd funtion in
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
# test_utility_funs.py (test_autograd_module_name)
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
