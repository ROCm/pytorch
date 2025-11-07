from __future__ import annotations

from textwrap import dedent

from .common import DeviceOpOverrides, register_device_op_overrides


class CpuDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

<<<<<<< HEAD
    def cpp_kernel_type(self) -> str:
        return "void*"

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def set_device(self, device_idx: int) -> str:
        return "pass"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: int) -> str:
        return "pass"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
