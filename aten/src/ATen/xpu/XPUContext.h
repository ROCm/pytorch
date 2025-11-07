#pragma once

#include <ATen/Context.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

// XPU is available if we compiled with XPU.
inline bool is_available() {
  return c10::xpu::device_count() > 0;
}

TORCH_XPU_API DeviceProp* getCurrentDeviceProperties();

TORCH_XPU_API DeviceProp* getDeviceProperties(DeviceIndex device);

TORCH_XPU_API int32_t getGlobalIdxFromDevice(DeviceIndex device);

<<<<<<< HEAD
TORCH_XPU_API bool canDeviceAccessPeer(DeviceIndex device, DeviceIndex peer);

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace at::xpu
