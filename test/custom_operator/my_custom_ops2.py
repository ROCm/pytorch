from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


<<<<<<< HEAD
@torch.library.register_fake("custom::sin")
=======
@torch.library.impl_abstract("custom::sin")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def sin_abstract(x):
    return torch.empty_like(x)
