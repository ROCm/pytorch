// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Installs the NCCL comm-registration hooks (see NCCLCommRegistrationHook.hpp)
// so that a backend which publishes a host ncclComm_t -- e.g. the nccl2
// ProcessGroupNCCL -- lands it in NCCLDevCommManager and participates in
// symmetric-memory lifecycle teardown without depending on the manager or
// allocator implementation. This is the one place the opaque void* comm from
// the hook is cast back to ncclComm_t.
//
// Gated on NCCL_HAS_SYMMEM_SUPPORT (via nccl_dev_cap.hpp), matching the other
// symm_mem NCCL translation units: the macro is only defined when NCCL is built
// with symmetric-memory support, so a build without it compiles this to an
// empty TU and producers' publish/retire calls stay no-ops.

#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <torch/csrc/distributed/c10d/NCCLCommRegistrationHook.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_cache.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#ifdef USE_ROCM
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/env.h>
#include <c10/util/Logging.h>

#include <chrono>
#include <exception>
#endif

namespace c10d::symmetric_memory {
namespace {

#ifdef USE_ROCM
bool prepareForGracefulRetirement(
    void* comm,
    c10::Device device,
    std::chrono::milliseconds timeout) {
  bool drained = false;
  const bool usedSymmMem =
      begin_symm_mem_teardown_for_comm(comm, timeout, &drained);
  if (!usedSymmMem) {
    return false;
  }
  if (!drained) {
    LOG(WARNING)
        << "Timed out waiting for symmetric-memory host operations before "
           "NCCL communicator teardown; retiring peer-allocation tables";
    return false;
  }

  try {
    c10::cuda::CUDAGuard guard(device);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    reclaim_retired_symm_mem_for_device(device);
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING)
        << "Failed to synchronize the device before symmetric-memory "
           "communicator teardown; retiring peer-allocation tables: "
        << e.what();
  } catch (...) {
    LOG(WARNING)
        << "Failed to synchronize the device before symmetric-memory "
           "communicator teardown; retiring peer-allocation tables";
  }
  return false;
}
#endif

// Installed at load time (before any process group is created).
const bool kNcclCommRegistrationHooksInstalled = [] {
  ::c10d::setNCCLCommRegistrationHooks(
      {// on_initialize
       [](void* comm) {
#ifdef USE_ROCM
         // RCCL samples these during communicator initialization. Record the
         // same state before publication, which may wait for a final group UID.
         note_rccl_symm_precondition(
             comm,
             c10::utils::check_env("NCCL_CUMEM_ENABLE") == true &&
                 c10::utils::check_env("NCCL_WIN_ENABLE") == true);
#else
         (void)comm;
#endif
       },
       // on_register
       [](const std::string& group_name, void* comm, c10::Device device) {
#ifdef USE_ROCM
         // Fallback for producers that publish without an initialization
         // notification. Conditional insertion preserves an earlier sample.
         note_rccl_symm_precondition(
             comm,
             c10::utils::check_env("NCCL_CUMEM_ENABLE") == true &&
                 c10::utils::check_env("NCCL_WIN_ENABLE") == true);
#endif
         NCCLDevCommManager::get(device).register_comm(
             group_name, reinterpret_cast<ncclComm_t>(comm));
       },
       // on_unregister
       [](const std::string& group_name,
          void* comm,
          c10::Device device,
          ::c10d::NCCLCommRetirementMode mode,
          std::chrono::milliseconds timeout) {
#ifdef USE_ROCM
         bool reclaimDeviceTables = false;
         if (mode == ::c10d::NCCLCommRetirementMode::Graceful) {
           reclaimDeviceTables =
               prepareForGracefulRetirement(comm, device, timeout);
         } else {
           bool drained = false;
           const bool usedSymmMem =
               begin_symm_mem_teardown_for_comm(comm, timeout, &drained);
           if (usedSymmMem && !drained) {
             LOG(WARNING)
                 << "Timed out waiting for symmetric-memory host operations "
                    "before aborting the NCCL communicator; its retired device "
                    "tables will remain quarantined";
           }
         }
#endif
         NCCLDevCommManager::get(device).unregister_comm(
             group_name, reinterpret_cast<ncclComm_t>(comm));
#ifdef USE_ROCM
         invalidate_symm_mem_for_comm(
             device, group_name, comm, reclaimDeviceTables);
         release_nccl_devcomms_for_group(device, group_name, comm);
         forget_rccl_symm_precondition(comm);
#endif
       }});
  return true;
}();

} // namespace
} // namespace c10d::symmetric_memory

#endif // NCCL_HAS_SYMMEM_SUPPORT
