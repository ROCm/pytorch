#pragma once

#include <atomic>

#include <ATen/Tensor.h>

namespace at {
namespace vulkan {

struct VulkanImplInterface {
  virtual ~VulkanImplInterface() = default;
  virtual bool is_vulkan_available() const = 0;
  virtual at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src)
      const = 0;
};

extern std::atomic<const VulkanImplInterface*> g_vulkan_impl_registry;

class VulkanImplRegistrar {
 public:
<<<<<<< HEAD
  explicit VulkanImplRegistrar(VulkanImplInterface* /*impl*/);
=======
  explicit VulkanImplRegistrar(VulkanImplInterface*);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
};

at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src);
} // namespace vulkan

namespace native {
  bool is_vulkan_available();
}// namespace native

} // namespace at
