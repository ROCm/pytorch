import functools
import logging
<<<<<<< HEAD
from typing import Optional

import torch
=======
import shutil
from typing import Optional

import torch
from torch._inductor.utils import clear_on_fresh_cache
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

from ... import config


log = logging.getLogger(__name__)


<<<<<<< HEAD
=======
@clear_on_fresh_cache
@functools.lru_cache(1)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def get_cuda_arch() -> Optional[str]:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception as e:
        log.error("Error getting cuda arch: %s", e)
        return None


<<<<<<< HEAD
=======
@clear_on_fresh_cache
@functools.lru_cache(1)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def get_cuda_version() -> Optional[str]:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        log.error("Error getting cuda version: %s", e)
        return None


<<<<<<< HEAD
@functools.lru_cache(None)
def nvcc_exist(nvcc_path: str = "nvcc") -> bool:
    if nvcc_path is None:
        return False
    import subprocess

    res = subprocess.call(
        ["which", nvcc_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return res == 0
=======
@functools.cache
def nvcc_exist(nvcc_path: Optional[str] = "nvcc") -> bool:
    return nvcc_path is not None and shutil.which(nvcc_path) is not None
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
