# PyTorch forward-mode is not mature yet
<<<<<<< HEAD
from torch._functorch.apis import chunk_vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
from torch.func import functionalize
=======
from functorch import functionalize
from torch._functorch.apis import chunk_vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import hessian, jacfwd, jvp
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
