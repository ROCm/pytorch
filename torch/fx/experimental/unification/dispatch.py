from functools import partial

<<<<<<< HEAD
from .multipledispatch import dispatch as _dispatch  # type: ignore[import]
=======
from .multipledispatch import dispatch  # type: ignore[import]
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


namespace = {}  # type: ignore[var-annotated]

<<<<<<< HEAD
dispatch = partial(_dispatch, namespace=namespace)
=======
dispatch = partial(dispatch, namespace=namespace)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
