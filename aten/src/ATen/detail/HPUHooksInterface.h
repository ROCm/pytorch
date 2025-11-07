#pragma once

#include <ATen/core/Generator.h>
#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/util/Registry.h>

namespace at {

struct TORCH_API HPUHooksInterface : AcceleratorHooksInterface {
  ~HPUHooksInterface() override = default;

  void init() const override {
    TORCH_CHECK(false, "Cannot initialize HPU without HPU backend");
  }

  virtual bool hasHPU() const {
    return false;
  }

  Device getDeviceFromPtr(void* /*data*/) const override {
    TORCH_CHECK(
        false, "Cannot get device of pointer on HPU without HPU backend");
  }

<<<<<<< HEAD
  bool isPinnedPtr(const void* /*data*/) const override {
=======
  bool isPinnedPtr(const void*) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    TORCH_CHECK(
        false,
        "You should register `HPUHooksInterface` for HPU before call `getPinnedMemoryAllocator`.");
  }

  bool hasPrimaryContext(
      [[maybe_unused]] DeviceIndex device_index) const override {
    TORCH_CHECK(
        false,
        "You should register `HPUHooksInterface` for HPU before call `hasPrimaryContext`.");
  }
};

struct TORCH_API HPUHooksArgs {};

TORCH_DECLARE_REGISTRY(HPUHooksRegistry, HPUHooksInterface, HPUHooksArgs);
#define REGISTER_HPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(HPUHooksRegistry, clsname, clsname)

namespace detail {

TORCH_API const at::HPUHooksInterface& getHPUHooks();

} // namespace detail
} // namespace at
