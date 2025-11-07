<<<<<<< HEAD
from typing import Any, Optional
=======
# mypy: allow-untyped-defs
from typing import Optional
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
)


@implements_per_sample_grads(F.embedding)
class EmbeddingPerSampleGrad(torch.autograd.Function):
    @staticmethod
<<<<<<< HEAD
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any, kwarg_names: list[str], _: Any, *expanded_args_and_kwargs: Any
    ) -> torch.Tensor:
=======
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        if len(expanded_args[0].shape) == 1:
            raise RuntimeError(
                f"Expanded Weights needs an input with a batch size, got a 1D tensor, {expanded_args[0]}"
            )
        output = forward_helper(F.embedding, expanded_args, expanded_kwargs)
        ctx.input, ctx.weight = expanded_args
        ctx.padding_idx, ctx.scale_grad_by_freq = (
            expanded_kwargs["padding_idx"],
            expanded_kwargs["scale_grad_by_freq"],
        )
        ctx.sparse = expanded_kwargs["sparse"]
        return output

    @staticmethod
<<<<<<< HEAD
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
=======
    def backward(ctx, grad_output):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        input, weight = ctx.input, ctx.weight
        padding_idx, scale_grad_by_freq, sparse = (
            ctx.padding_idx,
            ctx.scale_grad_by_freq,
            ctx.sparse,
        )

<<<<<<< HEAD
        def weight_per_sample_grad(weight: torch.Tensor) -> torch.Tensor:
=======
        def weight_per_sample_grad(weight):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            batch_size = input.shape[0]
            embedding_dim = weight.shape[1]
            index = (
                input.unsqueeze(-1)
                .expand(*input.shape, embedding_dim)
                .reshape(batch_size, -1, embedding_dim)
            )
<<<<<<< HEAD
            grad_sample = torch.zeros(  # type: ignore[attr-defined]
=======
            grad_sample = torch.zeros(
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype
            )
            return grad_sample.scatter_add_(
                1, index, grad_output.reshape(batch_size, -1, embedding_dim)
            )

        results: list[Optional[torch.Tensor]] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference

        if input.requires_grad:
            bw_fn = torch.ops.aten.embedding_backward
            results.append(
                bw_fn(
                    grad_output,
                    input,
                    weight.shape[0],
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )
            )
        else:
            results.append(None)

        # weight doesn't compute batched gradients; no other arguments are differentiable (2 not saved from forward)
        results = results + [None] * 6

        # set grad_sample field for weight with per sample gradients
        set_grad_sample_if_exists(weight, weight_per_sample_grad)
        return tuple(results)
