#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <torch/csrc/utils/pybind.h>
#if !defined(USE_ROCM)
#include <hip/hip_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif

#include <ATen/core/CachingHostAllocator.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>

namespace torch::cuda::shared {

#ifdef USE_ROCM
namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace
#endif

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

  py::enum_<hipError_t>(
      cudart,
      "cuda"
      "Error")
      .value("success", hipSuccess);

  cudart.def(
      "cuda"
      "GetErrorString",
      hipGetErrorString);
  cudart.def(
      "cuda"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      hipProfilerStart
#endif
  );
  cudart.def(
      "cuda"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      hipProfilerStop
#endif
  );
  cudart.def(
      "cuda"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> hipError_t {
        hipError_t err;
        {
          py::gil_scoped_release no_gil;
          err = C10_CUDA_ERROR_HANDLED(
              // NOLINTNEXTLINE(performance-no-int-to-ptr)
              hipHostRegister((void*)ptr, size, flags));
        }
        // Record after re-acquiring the GIL so that Python traceback
        // capture in maybeGatherContext() succeeds.
        if (err == hipSuccess) {
          if (auto* host_alloc = at::getHostAllocator(at::kCUDA)) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            host_alloc->record_external_register((void*)ptr, size);
          }
        }
        return err;
      });
  cudart.def(
      "cuda"
      "HostUnregister",
      [](uintptr_t ptr) -> hipError_t {
        hipError_t err;
        {
          py::gil_scoped_release no_gil;
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          err = C10_CUDA_ERROR_HANDLED(hipHostUnregister((void*)ptr));
        }
        if (err == hipSuccess) {
          if (auto* host_alloc = at::getHostAllocator(at::kCUDA)) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            host_alloc->record_external_unregister((void*)ptr);
          }
        }
        return err;
      });
  cudart.def(
      "cuda"
      "StreamCreate",
      [](uintptr_t ptr) -> hipError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_CUDA_ERROR_HANDLED(hipStreamCreate((hipStream_t*)ptr));
      });
  cudart.def(
      "cuda"
      "StreamDestroy",
      [](uintptr_t ptr) -> hipError_t {
        py::gil_scoped_release no_gil;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        return C10_CUDA_ERROR_HANDLED(hipStreamDestroy((hipStream_t)ptr));
      });
  cudart.def(
      "cuda"
      "MemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        c10::cuda::CUDAGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        py::gil_scoped_release no_gil;
        C10_CUDA_CHECK(hipMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::cuda::shared
