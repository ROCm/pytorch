#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif

#include <hip/hip_runtime.h>

#ifndef ROCM_ON_WINDOWS
#if CUDART_VERSION >= 13000 || defined(TORCH_CUDA_USE_NVTX3)
#include <roctracer/roctx.h>
#else // CUDART_VERSION >= 13000 || defined(TORCH_CUDA_USE_NVTX3)
#include <roctracer/roctx.h>
#endif // CUDART_VERSION >= 13000 || defined(TORCH_CUDA_USE_NVTX3)
#else // ROCM_ON_WINDOWS
#include <c10/util/Exception.h>
#endif // ROCM_ON_WINDOWS
#include <c10/hip/HIPException.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cuda::shared {

#ifndef ROCM_ON_WINDOWS
struct RangeHandle {
  int id;
  const char* msg;
};

static void device_callback_range_end(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  roctxRangeStop(handle->id);
  free((void*)handle->msg);
  free((void*)handle);
}

static void device_nvtxRangeEnd(void* handle, std::intptr_t stream) {
  C10_CUDA_CHECK(hipLaunchHostFunc(
      (hipStream_t)stream, device_callback_range_end, handle));
}

static void device_callback_range_start(void* userData) {
  RangeHandle* handle = ((RangeHandle*)userData);
  handle->id = roctxRangeStartA(handle->msg);
}

static void* device_nvtxRangeStart(const char* msg, std::intptr_t stream) {
  auto handle = static_cast<RangeHandle*>(calloc(1, sizeof(RangeHandle)));
  handle->msg = strdup(msg);
  handle->id = 0;
  TORCH_CHECK(
      hipLaunchHostFunc(
          (hipStream_t)stream, device_callback_range_start, (void*)handle) ==
      hipSuccess);
  return handle;
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#ifdef TORCH_CUDA_USE_NVTX3
  auto nvtx = m.def_submodule("_nvtx", "nvtx3 bindings");
#else
  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
#endif
  nvtx.def("rangePushA", roctxRangePushA);
  nvtx.def("rangePop", roctxRangePop);
  nvtx.def("rangeStartA", roctxRangeStartA);
  nvtx.def("rangeEnd", roctxRangeStop);
  nvtx.def("markA", roctxMarkA);
  nvtx.def("deviceRangeStart", device_nvtxRangeStart);
  nvtx.def("deviceRangeEnd", device_nvtxRangeEnd);
}

#else // ROCM_ON_WINDOWS

static void printUnavailableWarning() {
  TORCH_WARN_ONCE("Warning: roctracer isn't available on Windows");
}

static int rangePushA(const std::string&) {
  printUnavailableWarning();
  return 0;
}

static int rangePop() {
  printUnavailableWarning();
  return 0;
}

static int rangeStartA(const std::string&) {
  printUnavailableWarning();
  return 0;
}

static void rangeEnd(int) {
  printUnavailableWarning();
}

static void markA(const std::string&) {
  printUnavailableWarning();
}

static py::object deviceRangeStart(const std::string&, std::intptr_t) {
  printUnavailableWarning();
  return py::none(); // Return an appropriate default object
}

static void deviceRangeEnd(py::object, std::intptr_t) {
  printUnavailableWarning();
}

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto nvtx = m.def_submodule("_nvtx", "unavailable");

  nvtx.def("rangePushA", rangePushA);
  nvtx.def("rangePop", rangePop);
  nvtx.def("rangeStartA", rangeStartA);
  nvtx.def("rangeEnd", rangeEnd);
  nvtx.def("markA", markA);
  nvtx.def("deviceRangeStart", deviceRangeStart);
  nvtx.def("deviceRangeEnd", deviceRangeEnd);
}
#endif // ROCM_ON_WINDOWS

} // namespace torch::cuda::shared
