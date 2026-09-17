#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_CUFILE)
#include <c10/hip/HIPGuard.h>

#include <hip/hip_runtime.h>
#include <hipfile.h>

namespace {
// To get error message for hipFileRead/Write APIs that return ssize_t (-1 for
// filesystem error and a negative CUfileOpError enum value otherwise).
template <
    class T,
    std::enable_if_t<std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_HIPFILE_ERR(status) ? std::string(HIPFILE_ERRSTR(status))
                               : std::string(c10::utils::str_error(errno));
}

// To get error message for Buf/Handle registration APIs that return
// hipFileError_t
template <
    class T,
    std::enable_if_t<!std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_HIP_DRV_ERR(status))
    errStr.append(".").append(
        hipGetErrorString(static_cast<hipError_t>(status.hip_drv_err)));
  return errStr;
}
} // namespace

void gds_load_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  hipFileHandle_t cf_handle = reinterpret_cast<hipFileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Read the binary file
  ssize_t ret = hipFileRead(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "hipFileRead failed: ", cuGDSFileGetErrorString(ret));
}

void gds_save_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  hipFileHandle_t cf_handle = reinterpret_cast<hipFileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Write device memory contents to the file
  ssize_t ret = hipFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "hipFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

void gds_register_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  hipFileError_t status = hipFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(
      status.err == hipFileSuccess,
      "hipFileBufRegister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

void gds_deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  hipFileError_t status = hipFileBufDeregister(dataPtr);
  TORCH_CHECK(
      status.err == hipFileSuccess,
      "hipFileBufDeregister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

int64_t gds_register_handle(int fd) {
  hipFileDescr_t cf_descr;
  hipFileHandle_t cf_handle{};
  memset((void*)&cf_descr, 0, sizeof(hipFileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = hipFileHandleTypeOpaqueFD;
  hipFileError_t status = hipFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != hipFileSuccess) {
    TORCH_CHECK(
        false,
        "hipFileHandleRegister failed: ",
        cuGDSFileGetErrorString(status));
  }

  // Returning cuFileHandle_t as int64_t
  return reinterpret_cast<int64_t>(cf_handle);
}

void gds_deregister_handle(int64_t handle) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  hipFileHandle_t cf_handle = reinterpret_cast<hipFileHandle_t>(handle);
  hipFileHandleDeregister(cf_handle);
}

#endif

namespace torch::cuda::shared {

void initGdsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#if defined(USE_CUFILE)
  m.def("_gds_register_handle", &gds_register_handle);
  m.def("_gds_deregister_handle", &gds_deregister_handle);
  m.def("_gds_register_buffer", &gds_register_buffer);
  m.def("_gds_deregister_buffer", &gds_deregister_buffer);
  m.def("_gds_load_storage", &gds_load_storage);
  m.def("_gds_save_storage", &gds_save_storage);
#endif
}

} // namespace torch::cuda::shared
