#pragma once

#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

struct TORCH_API IPUHooksInterface : AcceleratorHooksInterface {
  ~IPUHooksInterface() override = default;

  void init() const override {
    TORCH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }

<<<<<<< HEAD
  bool hasPrimaryContext(DeviceIndex device_index) const override {
=======
  bool hasPrimaryContext(DeviceIndex /*device_index*/) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    TORCH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
    return false;
  }

  const Generator& getDefaultGenerator(
      [[maybe_unused]] DeviceIndex device_index = -1) const override {
    TORCH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }

  Generator getNewGenerator(
<<<<<<< HEAD
      DeviceIndex device_index [[maybe_unused]] = -1) const override {
=======
      DeviceIndex /*device_index*/ = -1) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    TORCH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }
};

struct TORCH_API IPUHooksArgs {};

TORCH_DECLARE_REGISTRY(IPUHooksRegistry, IPUHooksInterface, IPUHooksArgs);
#define REGISTER_IPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(IPUHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const IPUHooksInterface& getIPUHooks();
} // namespace detail
} // namespace at
