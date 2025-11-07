"""Internal flags for ONNX export."""

from __future__ import annotations

import functools
<<<<<<< HEAD
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable
=======
from typing import Any, Callable, cast, TypeVar
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


_is_onnx_exporting = False

<<<<<<< HEAD
# Use ParamSpec to preserve parameter types instead of erasing to Any
_P = ParamSpec("_P")
_R = TypeVar("_R")


def set_onnx_exporting_flag(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
=======
TCallable = TypeVar("TCallable", bound=Callable[..., Any])


def set_onnx_exporting_flag(func: TCallable) -> TCallable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        global _is_onnx_exporting
        _is_onnx_exporting = True
        try:
            return func(*args, **kwargs)
        finally:
            # Ensure it resets even if an exception occurs
            _is_onnx_exporting = False

<<<<<<< HEAD
    return wrapper
=======
    return cast(TCallable, wrapper)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
