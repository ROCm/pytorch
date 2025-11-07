from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


# NB: The impl_abstract_pystub for cos actually
# specifies it should live in the my_custom_ops2 module.
<<<<<<< HEAD
@torch.library.register_fake("custom::cos")
=======
@torch.library.impl_abstract("custom::cos")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def cos_abstract(x):
    return torch.empty_like(x)


# NB: There is no impl_abstract_pystub for tan
<<<<<<< HEAD
@torch.library.register_fake("custom::tan")
=======
@torch.library.impl_abstract("custom::tan")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def tan_abstract(x):
    return torch.empty_like(x)
