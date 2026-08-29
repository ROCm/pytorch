import torch

from ... import triton_utils as tu


# HIP rejects a launch once gridDim.x * blockDim.x reaches 2**32, and the
# rejection leaves the HIP context unusable rather than raising cleanly.
_HIP_MAX_LAUNCH_WORK_ITEMS = (1 << 32) - 1


def _is_outer_product(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[2] == 1
        and b.shape[1] == 1
        and a.numel() > 0
        and b.numel() > 0
        and not a.is_complex()
    )


def _bmm_outer_product_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    from .triton_kernels import bmm_outer_product

    device = a.get_device()
    if device == torch.cuda.current_device():
        return bmm_outer_product(a, b)

    with torch.cuda.device(device):
        return bmm_outer_product(a, b)


def _is_hip_grid_safe(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether the outer-product BMM launch is safe on this backend.

    Only HIP tensors are constrained; every other backend returns ``True``.
    Unlike Inductor, this eager kernel does not retune its fixed launch
    configuration, so oversized HIP launches decline the specialization and fall
    back to ATen. ATen can still fail on shapes this large, but it fails cleanly
    instead of leaving the HIP context unusable.
    """
    if torch.version.hip is None or a.device.type != "cuda":
        return True

    from .triton_kernels import (
        _bmm_outer_product_launch_config,
        _TRITON_DEFAULT_NUM_WARPS,
    )

    batch, m, _ = a.shape
    n = b.shape[2]
    grid_size, _, _ = _bmm_outer_product_launch_config(batch, m, n)
    warp_size = torch.cuda.get_device_properties(a.device).warp_size
    threads_per_program = _TRITON_DEFAULT_NUM_WARPS * warp_size
    return grid_size * threads_per_program <= _HIP_MAX_LAUNCH_WORK_ITEMS


def _bmm_outer_product_cond(
    a: torch.Tensor,
    b: torch.Tensor,
    *args,
    **kwargs,
) -> bool:
    a_is_cow = torch._C._is_cow_tensor(a)  # pyrefly: ignore[missing-attribute]
    b_is_cow = torch._C._is_cow_tensor(b)  # pyrefly: ignore[missing-attribute]
    return (
        a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and _is_outer_product(a, b)
        and not (a_is_cow or b_is_cow)
        and _is_hip_grid_safe(a, b)
    )


def _register_for_dispatch_key(dispatch_key: str) -> None:
    tu.register_op_override(
        "aten",
        "bmm",
        dispatch_key,
        cond=_bmm_outer_product_cond,
        impl=_bmm_outer_product_impl,
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    _register_for_dispatch_key("CUDA")
    if torch.xpu._is_compiled():
        _register_for_dispatch_key("XPU")
