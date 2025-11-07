#include <c10/core/impl/DeviceGuardImplInterface.h>
<<<<<<< HEAD
#include <c10/core/impl/FakeGuardImpl.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <array>

namespace c10::impl {

std::array<
    std::atomic<const DeviceGuardImplInterface*>,
    static_cast<size_t>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)>
    device_guard_impl_registry;

<<<<<<< HEAD
void registerDeviceGuard(
=======
DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    DeviceType type,
    const DeviceGuardImplInterface* impl) {
  device_guard_impl_registry[static_cast<size_t>(type)].store(impl);
}

<<<<<<< HEAD
DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(
    DeviceType type,
    const DeviceGuardImplInterface* impl) {
  registerDeviceGuard(type, impl);
}

namespace {
thread_local std::unique_ptr<DeviceGuardImplInterface> tls_fake_device_guard =
    nullptr;
} // namespace

void ensureCUDADeviceGuardSet() {
  constexpr auto cuda_idx = static_cast<std::size_t>(DeviceType::CUDA);

  const DeviceGuardImplInterface* p =
      device_guard_impl_registry[cuda_idx].load();

  // A non-null `ptr` indicates that the CUDA guard is already set up,
  // implying this is using cuda build
  if (p && p->deviceCount() == 0) {
    // In following cases, we override CUDA guard interface with a no-op
    // device guard. When p->deviceCount() == 0, cuda build is enabled, but no
    // cuda devices available.
    tls_fake_device_guard = std::make_unique<FakeGuardImpl<DeviceType::CUDA>>();
    device_guard_impl_registry[cuda_idx].store(tls_fake_device_guard.get());
  }
}

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace c10::impl
