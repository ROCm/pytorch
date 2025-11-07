import dataclasses
<<<<<<< HEAD
from collections.abc import Callable
from typing import Optional
=======
from typing import Callable, Optional
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


all_experiments: dict[str, Callable] = {}


@dataclasses.dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float
    dtype: str
    device: str
    arch: str  # GPU name for CUDA or CPU arch for CPU
    is_model: bool = False


def register_experiment(name: Optional[str] = None):
    def decorator(func):
        key = name or func.__name__
        all_experiments[key] = func
        return func

    return decorator
