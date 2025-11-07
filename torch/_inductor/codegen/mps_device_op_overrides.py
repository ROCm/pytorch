from __future__ import annotations

from .common import DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    def device_guard(self, device_idx: int) -> str:
        assert device_idx == 0
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: int) -> str:
        assert device_idx == 0
<<<<<<< HEAD
        return "# MPS set device"
=======
        return "pass  # MPS set device"

    def kernel_driver(self) -> str:
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


register_device_op_overrides("mps", MPSDeviceOpOverrides())
