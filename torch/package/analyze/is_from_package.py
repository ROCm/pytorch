from types import ModuleType
from typing import Any

from .._mangling import is_mangled


def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
<<<<<<< HEAD
    if type(obj) is ModuleType:
=======
    if type(obj) == ModuleType:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return is_mangled(obj.__name__)
    else:
        return is_mangled(type(obj).__module__)
