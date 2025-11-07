#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <c10/core/Stream.h>
#include <c10/util/Registry.h>

#include <c10/core/Allocator.h>

#include <c10/util/python_stub.h>
#include <ATen/detail/AcceleratorHooksInterface.h>

#include <string>
<<<<<<< HEAD
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
namespace at {
class Context;
}

namespace at {
constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

struct TORCH_API MTIAHooksInterface : AcceleratorHooksInterface {
// this fails the implementation if MTIAHooks functions are called, but
// MTIA backend is not present.
#define FAIL_MTIAHOOKS_FUNC(func) \
  TORCH_CHECK(false, "Cannot execute ", func, "() without MTIA backend.");

  ~MTIAHooksInterface() override = default;

  void init() const override {
    // Avoid logging here, since MTIA needs init devices first then it will know
    // how many devices are available. Make it as no-op if mtia extension is not
    // dynamically loaded.
    return;
  }

  virtual bool hasMTIA() const {
    return false;
  }

  DeviceIndex deviceCount() const override {
    return 0;
  }

<<<<<<< HEAD
  virtual void deviceSynchronize(c10::DeviceIndex device_index) const {
=======
  virtual void deviceSynchronize(c10::DeviceIndex /*device_index*/) const {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual std::string showConfig() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

<<<<<<< HEAD
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    return false;
  }

  void setCurrentDevice(DeviceIndex device) const override {
=======
  bool hasPrimaryContext(DeviceIndex /*device_index*/) const override {
    return false;
  }

  void setCurrentDevice(DeviceIndex /*device*/) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  DeviceIndex getCurrentDevice() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

<<<<<<< HEAD
  DeviceIndex exchangeDevice(DeviceIndex device) const override {
=======
  DeviceIndex exchangeDevice(DeviceIndex /*device*/) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

<<<<<<< HEAD
  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {
=======
  DeviceIndex maybeExchangeDevice(DeviceIndex /*device*/) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

<<<<<<< HEAD
  virtual c10::Stream getCurrentStream(DeviceIndex device) const {
=======
  virtual c10::Stream getCurrentStream(DeviceIndex /*device*/) const {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

<<<<<<< HEAD
  virtual c10::Stream getDefaultStream(DeviceIndex device) const {
=======
  virtual int64_t getCurrentRawStream(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  virtual c10::Stream getDefaultStream(DeviceIndex /*device*/) const {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

<<<<<<< HEAD
  virtual void setCurrentStream(const c10::Stream& stream) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  bool isPinnedPtr(const void* data) const override {
=======
  virtual void setCurrentStream(const c10::Stream& /*stream*/ ) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  bool isPinnedPtr(const void* /*data*/) const override {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

<<<<<<< HEAD
  virtual PyObject* memoryStats(DeviceIndex device) const {
=======
  virtual PyObject* memoryStats(DeviceIndex /*device*/) const {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

<<<<<<< HEAD
  virtual PyObject* getDeviceCapability(DeviceIndex device) const {
=======
  virtual PyObject* getDeviceCapability(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual PyObject* getDeviceProperties(DeviceIndex device) const {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual void emptyCache() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }


  virtual void recordMemoryHistory(
<<<<<<< HEAD
    const std::optional<std::string>& enabled,
    const std::string& stacks,
    size_t max_entries) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual PyObject* memorySnapshot() const {
=======
    const std::optional<std::string>& /*enabled*/,
    const std::string& /*stacks*/,
    size_t /*max_entries*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual PyObject* memorySnapshot(const std::optional<std::string>& local_path) const {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    FAIL_MTIAHOOKS_FUNC(__func__);
    return nullptr;
  }

  virtual DeviceIndex getDeviceCount() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return 0;
  }

<<<<<<< HEAD
  virtual void resetPeakMemoryStats(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

=======
  virtual void resetPeakMemoryStats(DeviceIndex /*device*/) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  virtual void attachOutOfMemoryObserver(PyObject* observer) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return;
  }
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
};

struct TORCH_API MTIAHooksArgs {};

TORCH_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);
#define REGISTER_MTIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MTIAHooksInterface& getMTIAHooks();
TORCH_API bool isMTIAHooksBuilt();
} // namespace detail
} // namespace at
<<<<<<< HEAD
C10_DIAGNOSTIC_POP()
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
