#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/UvmMemoryAllocator.h>
#include <c10/core/Storage.h>

namespace at {

namespace native {

bool is_managed_cuda(const Tensor& self, c10::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (!device.has_value() && device->is_cpu()) || device->is_cuda());
  // TODO: unhook this
  return detail::getCUDAHooks().isManagedPtr(self.storage().data());
}

Tensor _manage_memory_cuda(const Tensor& self, c10::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (!device.has_value() && device->is_cpu()) || device->is_cuda());
  at::Allocator* allocator = nullptr;
  if (self.is_cpu()) {
    allocator = at::cuda::getUnifiedDeviceAllocatorCpu();
  } else if (self.is_cuda()) {
    allocator = at::cuda::getUnifiedDeviceAllocator();
  }
  size_t size_bytes = detail::computeStorageNbytes(
      self.sizes(), self.strides(), self.dtype().itemsize());
  auto storage = Storage(
      Storage::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizable=*/false);
  auto tensor = at::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace native
} // namespace at
