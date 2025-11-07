"""This module converts objects into numpy array."""

import numpy as np

import torch


def make_np(x: torch.Tensor) -> np.ndarray:
    """
    Convert an object into numpy array.

    Args:
      x: An instance of torch tensor

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
<<<<<<< HEAD
        if x.device.type == "meta":
            return np.random.randn(1)
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return _prepare_pytorch(x)
    raise NotImplementedError(
        f"Got {type(x)}, but numpy array or torch tensor are expected."
    )


def _prepare_pytorch(x: torch.Tensor) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float16)
<<<<<<< HEAD
    # pyrefly: ignore [bad-assignment]
    x = x.detach().cpu().numpy()
    # pyrefly: ignore [bad-return]
=======
    x = x.detach().cpu().numpy()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    return x
