#include <ATen/LegacyVmapMode.h>

namespace at::impl {

<<<<<<< HEAD
thread_local int64_t VmapMode_current_vmap_level = 0;
=======
thread_local static int64_t VmapMode_current_vmap_level = 0;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

int64_t VmapMode::current_vmap_level() {
  return VmapMode_current_vmap_level;
}

int64_t VmapMode::increment_nesting() {
  VmapMode_current_vmap_level++;
  if (VmapMode_current_vmap_level == 1) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, true);
  }
  return VmapMode_current_vmap_level;
}

int64_t VmapMode::decrement_nesting() {
  VmapMode_current_vmap_level--;
  if (VmapMode_current_vmap_level == 0) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, false);
  }
  return VmapMode_current_vmap_level;
}
} // namespace at::impl
