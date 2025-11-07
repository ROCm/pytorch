#include <torch/csrc/lazy/core/config.h>
<<<<<<< HEAD

// TODO(whc) unclear if this is useful, has only been tested as true
// NOLINTNEXTLINE(misc-use-internal-linkage)
=======
#include <torch/csrc/lazy/ts_backend/config.h>

// TODO(whc) unclear if this is useful, has only been tested as true
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
C10_DEFINE_bool(
    torch_lazy_ts_tensor_update_sync,
    true,
    "Use synchronous copy inside _copy_from op")

// TODO(whc) we need to hook up these flags in a more useful way
// possibly also keep LTC_TS_CUDA env working?
<<<<<<< HEAD
// NOLINTNEXTLINE(misc-use-internal-linkage)
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
C10_DEFINE_bool(
    torch_lazy_ts_cuda,
    false,
    "Use cuda device for torchscript backend (instead of CPU)")
