#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <ATen/hipdnn/Handle.h>
#include <c10/hip/HIPStream.h>

#include <ATen/hip/Exceptions.h>
#include <ATen/hipdnn/Exceptions.h>
// #include <hipdnn_frontend.hpp>

namespace at::native {
namespace {

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

void createHipdnnHandle(hipdnnHandle_t *handle) {
  HIPDNN_CHECK(hipdnnBackend()->create(handle));
}

void destroyHipdnnHandle(hipdnnHandle_t handle) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the cuda context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. @colesbury and I decided to not destroy
  // the handle as a workaround.
  //   - @soumith
  //
  // Further note: this is now disabled globally, because we are seeing
  // the same issue as mentioned above in CUDA 11 CI.
  //   - @zasdfgbnm
  //
  // #ifdef NO_MIOPEN_DESTROY_HANDLE
  // #else
  //   miopenDestroy(handle);
  // #endif
}

using HipdnnPoolType = at::cuda::DeviceThreadHandlePool<
    hipdnnHandle_t,
    createHipdnnHandle,
    destroyHipdnnHandle>;

} // namespace

hipdnnHandle_t getHipdnnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::hip::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<HipdnnPoolType>();
  thread_local std::unique_ptr<HipdnnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  HIPDNN_CHECK(hipdnnBackend()->setStream(handle, c10::hip::getCurrentHIPStream()));
  return handle;
}

} // namespace at::native
