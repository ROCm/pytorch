from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


<<<<<<< HEAD
@torch.library.register_fake("custom::nonzero")
=======
@torch.library.impl_abstract("custom::nonzero")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def nonzero_abstract(x):
    n = x.dim()
    ctx = torch.library.get_ctx()
    nnz = ctx.create_unbacked_symint()
    shape = [nnz, n]
    return x.new_empty(shape, dtype=torch.long)
