from __future__ import annotations

import functools
<<<<<<< HEAD
=======
import linecache
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
<<<<<<< HEAD
from typing import Callable, TYPE_CHECKING
=======
from typing import Any, Callable, TYPE_CHECKING
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


if TYPE_CHECKING:
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


<<<<<<< HEAD
def _reload_python_module_in_subproc(key: str, path: str) -> ModuleType:
    codecache = sys.modules.get("torch._inductor.codecache")
    if codecache:
        return codecache.PyCodeCache.load_by_key_path(key, path)
    else:
        return _reload_python_module(key, path)


def _reload_python_module(key: str, path: str) -> ModuleType:
=======
def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
<<<<<<< HEAD
        sys.modules[mod.__name__] = mod
        return mod


@functools.lru_cache(None)
=======
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[1] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _worker_compile_triton(
<<<<<<< HEAD
    load_kernel: Callable[[], CachingAutotuner], extra_env: dict[str, str]
) -> tuple[CachingAutotuner, int]:
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    start_ns = time.time_ns()
    kernel = load_kernel()
    kernel.precompile(warm_cache_only=True)
    elapsed_ns = time.time_ns() - start_ns
    kernel.prepare_for_pickle()
    return kernel, elapsed_ns // 1000
=======
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str],
    extra_config: dict[str, Any],
) -> tuple[CachingAutotuner, int]:
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    from torch._inductor import config

    with config.patch(extra_config):
        start_ns = time.time_ns()
        kernel = load_kernel()
        kernel.precompile(warm_cache_only=True)
        elapsed_ns = time.time_ns() - start_ns
        kernel.prepare_for_pickle()
        # We can release this memory in the compile subprocesses:
        linecache.clearcache()
        return kernel, elapsed_ns // 1000
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
