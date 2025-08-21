from typing import List, Dict, Tuple, Any
import time

_torch_module = None
def _get_torch_or_raise() -> Any: # Changed return type to Any for flexibility
    """Lazily imports and returns the torch module."""
    global _torch_module
    if _torch_module is None:
        try:
            import torch
            _torch_module = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for EventReplayer functionality that is being used. "
                "Please install PyTorch."
            )
    return _torch_module

list_profile_tensor_types = ['double', 'float', 'c10::Half', 'c10::BFloat16',
                             'long', 'int', 'bool']

from dataclasses import dataclass
@dataclass
class TensorCfg:
    """
    A class to represent a dummy tensor.
    """
    shape: List[int]
    dtype: str
    strides: List[int]
    init: str = "normal"


def build_tensor(cfg: TensorCfg, device: str='cuda') -> 'torch.Tensor':

    torch = _get_torch_or_raise()
    dict_profile2torchdtype = {
        'bool': torch.bool,
        'int': torch.int,
        'long': torch.long,
        'double': torch.float64,
        'float': torch.float32,
        'c10::Half': torch.float16,
        'c10::BFloat16': torch.bfloat16,
    }
    dtype  = dict_profile2torchdtype[cfg.dtype]
    size   = cfg.shape
    stride = cfg.strides
    # allocate *exactly* the storage needed for that stride/shape
    t = torch.empty_strided(size, stride, dtype=dtype, device=device)
    is_floating = t.is_floating_point() or t.is_complex()
    init = cfg.init
    if init == 'normal':
        if not is_floating:
            raise ValueError(f"Cannot initialize tensor of type {cfg.dtype} with 'normal' init.")
        t.normal_()                     # or whatever init you like
    elif init is not None:
        raise ValueError(f"Unsupported tensor initialization: {init}")
    return t

def summarize_tensor(tensor: 'torch.Tensor') -> str:
    """
    Summarize the tensor information.
    
    Args:
        tensor (torch.Tensor): The tensor to summarize.
    
    Returns:
        str: The summary string.
    """
    return f"Tensor(shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, strides={tensor.stride()})"

def benchmark_func(func, device, warmup=50, avg_steps=100):
    """
    Benchmark a function with warmup and average steps.
    Disclaimer: This method would be inaccurate for very short ops.
    Args:
        func (callable): The function to benchmark.
        warmup (int): Number of warmup iterations.
        avg_steps (int): Number of iterations to average over.
    Returns:
        float: Average time taken per iteration in microseconds.
    """
    torch = _get_torch_or_raise()
    # Warmup phase
    for _ in range(warmup):
        func()

    # Benchmarking phase
    torch.cuda.synchronize(device)
    start_time = time.time()
    for _ in range(avg_steps):
        func()
    torch.cuda.synchronize(device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_sec = elapsed_time / avg_steps
    avg_time_us = avg_time_sec * 1e6

    return avg_time_us