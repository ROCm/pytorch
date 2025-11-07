<<<<<<< HEAD
=======
# mypy: allow-untyped-defs
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from typing import Optional

import torch


__all__ = [
    "version",
    "is_available",
    "get_max_alg_id",
]

try:
    from torch._C import _cusparselt
except ImportError:
    _cusparselt = None  # type: ignore[assignment]

__cusparselt_version: Optional[int] = None
__MAX_ALG_ID: Optional[int] = None

if _cusparselt is not None:

<<<<<<< HEAD
    def _init() -> bool:
        global __cusparselt_version
        global __MAX_ALG_ID
        if __cusparselt_version is None:
            # pyrefly: ignore [missing-attribute]
=======
    def _init():
        global __cusparselt_version
        global __MAX_ALG_ID
        if __cusparselt_version is None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            __cusparselt_version = _cusparselt.getVersionInt()
            if __cusparselt_version == 400:
                __MAX_ALG_ID = 4
            elif __cusparselt_version == 502:
                __MAX_ALG_ID = 5
            elif __cusparselt_version == 602:
                __MAX_ALG_ID = 37
        return True

else:

<<<<<<< HEAD
    def _init() -> bool:
=======
    def _init():
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return False


def version() -> Optional[int]:
    """Return the version of cuSPARSELt"""
    if not _init():
        return None
    return __cusparselt_version


def is_available() -> bool:
    r"""Return a bool indicating if cuSPARSELt is currently available."""
    return torch._C._has_cusparselt


def get_max_alg_id() -> Optional[int]:
    if not _init():
        return None
    return __MAX_ALG_ID
