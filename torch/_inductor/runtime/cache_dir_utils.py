import getpass
import os
import re
import tempfile
<<<<<<< HEAD
=======
from collections.abc import Generator
from contextlib import contextmanager

from torch._environment import is_fbcode
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


# Factoring out to file without torch dependencies


def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_cache_dir() -> str:
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return os.path.join(
<<<<<<< HEAD
        tempfile.gettempdir(),
=======
        tempfile.gettempdir() if not is_fbcode() else "/var/tmp",
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        "torchinductor_" + sanitized_username,
    )


def triton_cache_dir(device: int) -> str:
    if (directory := os.getenv("TRITON_CACHE_DIR")) is not None:
        return directory
    return os.path.join(
        cache_dir(),
        "triton",
        str(device),
    )
<<<<<<< HEAD
=======


@contextmanager
def temporary_cache_dir(directory: str) -> Generator[None, None, None]:
    from torch._inductor.utils import clear_caches

    original = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = directory
    try:
        clear_caches()
        yield
    finally:
        clear_caches()
        if original is None:
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = original
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
