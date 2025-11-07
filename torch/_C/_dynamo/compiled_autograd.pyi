from typing import Callable

<<<<<<< HEAD
=======
from torch import Tensor
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from torch._dynamo.compiled_autograd import AutogradCompilerInstance

def set_autograd_compiler(
    autograd_compiler: Callable[[], AutogradCompilerInstance] | None,
    dynamic: bool,
) -> tuple[Callable[[], AutogradCompilerInstance] | None, bool]: ...
def clear_cache() -> None: ...
def is_cache_empty() -> bool: ...
def set_verbose_logger(fn: Callable[[str], None] | None) -> bool: ...
<<<<<<< HEAD
=======
def call_cpp_tensor_pre_hooks(idx: int, grad: Tensor) -> Tensor: ...
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
