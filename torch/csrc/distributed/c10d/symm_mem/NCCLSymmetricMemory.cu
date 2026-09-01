#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <algorithm>
#include <atomic>
#include <iterator>
#include <vector_types.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/env.h>
#include <c10/util/error.h>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>

#ifndef NCCL_WIN_REQUIRED_ALIGNMENT
#define NCCL_WIN_REQUIRED_ALIGNMENT 4096
#endif

namespace c10d {
namespace symmetric_memory {

#ifdef USE_ROCM
// Byte budget for the ROCm free-block cache (see NCCLSymmetricMemoryAllocator).
constexpr size_t kFreeCacheByteBudget = 128UL * 1024 * 1024;
#endif

// This owned device-communicator cache exists only for ROCm: RCCL's
// <nccl_device.h> is not host-compilable, so NCCLDevCommManager (a host-included
// header) cannot store ncclDevComm on ROCm. On CUDA that header is host-safe, so
// the ops use NCCLDevCommManager directly and this cache is never compiled in.
// Keep CUDA on the stub and off ncclDevCommCreate entirely.
#if defined(USE_ROCM) && defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT)
namespace {

// Device communicators are owned here rather than in NCCLDevCommManager
// because this is the only TU that can name ncclDevComm on ROCm (RCCL's
// device header does not survive host-only compiles). Keyed by
// (device, group, key); entries die with the owning process group via
// release_nccl_devcomms_for_group, so a recreated group can never observe
// a communicator built for its predecessor. Survivors are destroyed at
// process exit, mirroring ~NCCLDevCommManager.
struct NcclDevCommCache {
  struct Entry {
    ncclDevComm devcomm{};
    ncclComm_t owner{nullptr};
  };
  struct GroupState {
    ska::flat_hash_map<std::string, Entry> by_key;
    std::mutex mutex;
  };
  ska::flat_hash_map<
      int,
      ska::flat_hash_map<std::string, std::shared_ptr<GroupState>>>
      by_device;
  std::mutex mutex;

  ~NcclDevCommCache() {
    if (is_finalizing()) {
      return;
    }
    for (auto& [dev_idx, groups] : by_device) {
      try {
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(dev_idx));
        // No kernel may still be using a communicator when it is destroyed.
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        for (auto& group_entry : groups) {
          auto& state = group_entry.second;
          if (!state) {
            continue;
          }
          std::lock_guard<std::mutex> lock(state->mutex);
          for (auto& key_entry : state->by_key) {
            ncclDevCommDestroy(
                key_entry.second.owner, &key_entry.second.devcomm);
          }
        }
      } catch (...) {
        LOG(WARNING) << "Failed to destroy NCCL device communicators, skipping";
      }
    }
  }
};

NcclDevCommCache& devcomm_cache() {
  static NcclDevCommCache cache;
  return cache;
}

} // namespace

void get_or_create_nccl_devcomm(
    const c10::Device& device,
    const std::string& group_name,
    const std::string& key,
    int lsa_barrier_count,
    bool lsa_multimem,
    void* devcomm_out,
    size_t devcomm_size) {
  TORCH_CHECK(devcomm_out != nullptr, "devcomm_out must not be null");
  TORCH_CHECK(
      devcomm_size == sizeof(ncclDevComm),
      "Unexpected ncclDevComm size: expected ",
      sizeof(ncclDevComm),
      ", got ",
      devcomm_size);
  ncclComm_t comm =
      NCCLDevCommManager::get(device).get_comm(group_name);
  auto& cache = devcomm_cache();
  std::shared_ptr<NcclDevCommCache::GroupState> state;
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto& state_ref = cache.by_device[device.index()][group_name];
    if (!state_ref) {
      state_ref = std::make_shared<NcclDevCommCache::GroupState>();
    }
    state = state_ref;
  }
  std::lock_guard<std::mutex> state_lock(state->mutex);
  auto* live_comm = NCCLDevCommManager::get(device).get_comm(group_name);
  TORCH_CHECK(
      live_comm == comm,
      "The process-group communicator changed while creating an NCCL device "
      "communicator. Retry the operation.");
  auto& entry = state->by_key[key];
  if (entry.owner != comm) {
    // entry.owner == nullptr: first use for this (device, group, key).
    // entry.owner != nullptr: a predecessor process group's stale devcomm
    //   survived because its destructor was skipped (the host comm had
    //   already aborted, so ~ProcessGroupNCCL took the isAborted() continue
    //   and never released it). Rebuild for the current comm rather than
    //   asserting. Do not ncclDevCommDestroy the stale one -- the aborted
    //   comm reclaims its own resources (same rationale as release below).
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = lsa_barrier_count;
    reqs.lsaMultimem = lsa_multimem;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &entry.devcomm),
        "ncclDevCommCreate failed");
    entry.owner = comm;
  }
  *static_cast<ncclDevComm*>(devcomm_out) = entry.devcomm;
}

void release_nccl_devcomms_for_group(
    const c10::Device& device,
    const std::string& group_name,
    void* comm) {
  auto* owner = static_cast<ncclComm_t>(comm);
  auto& cache = devcomm_cache();
  // Keep the top-level entry locked until its state has been examined and, if
  // empty, removed. Otherwise a successor can capture `state`, the predecessor
  // can detach it from by_device, and the successor can then insert into an
  // untracked state that teardown will never see.
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto dev_it = cache.by_device.find(device.index());
  if (dev_it == cache.by_device.end()) {
    return;
  }
  auto group_it = dev_it->second.find(group_name);
  if (group_it == dev_it->second.end()) {
    return;
  }
  auto state = group_it->second;
  if (!state) {
    return;
  }
  // Identity-safe: erase only entries this comm owns. A stale destructor whose
  // comm was already replaced by a successor under the same group name (e.g.
  // restart-after-error) leaves the successor's entries untouched. Erase
  // without ncclDevCommDestroy: kernels from other streams may still reference
  // the communicator at teardown time; the owning comm reclaims the resources,
  // and only process-exit survivors need explicit destruction (see
  // ~NcclDevCommCache).
  {
    std::lock_guard<std::mutex> state_lock(state->mutex);
    for (auto it = state->by_key.begin(); it != state->by_key.end();) {
      if (it->second.owner == owner) {
        it = state->by_key.erase(it);
      } else {
        ++it;
      }
    }
    if (state->by_key.empty()) {
      dev_it->second.erase(group_it);
    }
  }
}
#else

void get_or_create_nccl_devcomm(
    const c10::Device& /*device*/,
    const std::string& /*group_name*/,
    const std::string& /*key*/,
    int /*lsa_barrier_count*/,
    bool /*lsa_multimem*/,
    void* /*devcomm_out*/,
    size_t /*devcomm_size*/) {
  TORCH_CHECK(
      false,
      "NCCL device communicators via the owned cache are ROCm-only "
      "(RCCL >= 2.29.7); CUDA uses NCCLDevCommManager.");
}

void release_nccl_devcomms_for_group(
    const c10::Device&,
    const std::string&,
    void*) {}

#endif // USE_ROCM && NCCL_HAS_SYMMEM_DEVICE_SUPPORT

#ifdef USE_ROCM
namespace {
// Snapshot of the RCCL window preconditions (NCCL_CUMEM_ENABLE /
// NCCL_WIN_ENABLE) as they were when a host comm was created -- the moment RCCL
// actually sampled them in ncclCommInitRank. Keyed by the host ncclComm_t (not
// by group name) so rendezvous looks it up by the comm it resolves: a later
// comm reusing a group name gets its own (or no) entry and never inherits a
// destroyed comm's value.
struct RcclSymmPreconditionMap {
  ska::flat_hash_map<ncclComm_t, bool> by_comm;
  std::mutex mutex;
};

void open_symm_mem_lifecycle(ncclComm_t comm);

RcclSymmPreconditionMap& rccl_symm_precondition_map() {
  static RcclSymmPreconditionMap m;
  return m;
}

std::optional<bool> rccl_symm_precondition_lookup(ncclComm_t comm) {
  auto& m = rccl_symm_precondition_map();
  std::lock_guard<std::mutex> lock(m.mutex);
  auto it = m.by_comm.find(comm);
  if (it == m.by_comm.end()) {
    return std::nullopt;
  }
  return it->second;
}
} // namespace

void note_rccl_symm_precondition(void* comm, bool ok) {
  open_symm_mem_lifecycle(static_cast<ncclComm_t>(comm));
  auto& m = rccl_symm_precondition_map();
  std::lock_guard<std::mutex> lock(m.mutex);
  // A split/shrink child inherits the parent's init-time configuration before
  // its final group UID is assigned and publication runs. Preserve that
  // inherited snapshot instead of replacing it with the current environment.
  auto* nccl_comm = static_cast<ncclComm_t>(comm);
  if (m.by_comm.find(nccl_comm) == m.by_comm.end()) {
    m.by_comm.emplace(nccl_comm, ok);
  }
}

void inherit_rccl_symm_precondition(void* parent_comm, void* child_comm) {
  auto& m = rccl_symm_precondition_map();
  std::lock_guard<std::mutex> lock(m.mutex);
  auto parent_it = m.by_comm.find(static_cast<ncclComm_t>(parent_comm));
  if (parent_it == m.by_comm.end()) {
    return;
  }
  m.by_comm[static_cast<ncclComm_t>(child_comm)] = parent_it->second;
}

void forget_rccl_symm_precondition(void* comm) {
  auto& m = rccl_symm_precondition_map();
  std::lock_guard<std::mutex> lock(m.mutex);
  m.by_comm.erase(static_cast<ncclComm_t>(comm));
}
#else
void note_rccl_symm_precondition(void*, bool) {}
void inherit_rccl_symm_precondition(void*, void*) {}
void forget_rccl_symm_precondition(void*) {}
#endif // USE_ROCM

NCCLSymmetricMemoryLaunchGuard::NCCLSymmetricMemoryLaunchGuard(
    std::function<void()> release)
    : release_(std::move(release)) {}

NCCLSymmetricMemoryLaunchGuard::NCCLSymmetricMemoryLaunchGuard(
    NCCLSymmetricMemoryLaunchGuard&& other) noexcept
    : release_(std::move(other.release_)) {
  other.release_ = nullptr;
}

NCCLSymmetricMemoryLaunchGuard& NCCLSymmetricMemoryLaunchGuard::operator=(
    NCCLSymmetricMemoryLaunchGuard&& other) noexcept {
  if (this != &other) {
    releaseNoexcept();
    release_ = std::move(other.release_);
    other.release_ = nullptr;
  }
  return *this;
}

void NCCLSymmetricMemoryLaunchGuard::releaseNoexcept() noexcept {
  if (release_) {
    try {
      release_();
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to release NCCL symmetric-memory launch guard: "
                   << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to release NCCL symmetric-memory launch guard";
    }
    release_ = nullptr;
  }
}

NCCLSymmetricMemoryLaunchGuard::~NCCLSymmetricMemoryLaunchGuard() {
  releaseNoexcept();
}

#ifdef USE_ROCM
namespace {

struct SymmMemCommLifecycle {
  std::mutex mutex;
  std::condition_variable cv;
  bool closing = false;
  bool used = false;
  // Once a teardown timed out without draining admitted host enqueuers, a
  // later device synchronization cannot prove it ran after those enqueues.
  // Keep their retired device tables alive for the rest of the process.
  bool reclaimable_after_sync = true;
  size_t active_launches = 0;
};

struct SymmMemCommLifecycleRegistry {
  ska::flat_hash_map<ncclComm_t, std::shared_ptr<SymmMemCommLifecycle>> by_comm;
  std::mutex mutex;
};

SymmMemCommLifecycleRegistry& symm_mem_lifecycle_registry() {
  static SymmMemCommLifecycleRegistry registry;
  return registry;
}

std::shared_ptr<SymmMemCommLifecycle> get_symm_mem_lifecycle(ncclComm_t comm) {
  auto& registry = symm_mem_lifecycle_registry();
  std::lock_guard<std::mutex> registry_lock(registry.mutex);
  auto& lifecycle = registry.by_comm[comm];
  if (!lifecycle) {
    lifecycle = std::make_shared<SymmMemCommLifecycle>();
  }
  return lifecycle;
}

std::shared_ptr<SymmMemCommLifecycle> find_symm_mem_lifecycle(ncclComm_t comm) {
  auto& registry = symm_mem_lifecycle_registry();
  std::lock_guard<std::mutex> registry_lock(registry.mutex);
  auto it = registry.by_comm.find(comm);
  if (it == registry.by_comm.end()) {
    return nullptr;
  }
  return it->second;
}

void open_symm_mem_lifecycle(ncclComm_t comm) {
  (void)get_symm_mem_lifecycle(comm);
}

void forget_symm_mem_lifecycle(ncclComm_t comm) {
  auto& registry = symm_mem_lifecycle_registry();
  std::lock_guard<std::mutex> registry_lock(registry.mutex);
  registry.by_comm.erase(comm);
}

NCCLSymmetricMemoryLaunchGuard acquire_symm_mem_launch_guard(
    const std::shared_ptr<SymmMemCommLifecycle>& lifecycle) {
  TORCH_INTERNAL_ASSERT(lifecycle != nullptr);
  {
    std::unique_lock<std::mutex> lock(lifecycle->mutex);
    TORCH_CHECK(
        !lifecycle->closing,
        "This symmetric-memory handle is bound to a destroyed communicator or "
        "freed backing allocation. "
        "Rendezvous the tensor again with a live process group.");
    lifecycle->used = true;
    ++lifecycle->active_launches;
  }
  return NCCLSymmetricMemoryLaunchGuard([lifecycle]() {
    std::lock_guard<std::mutex> lock(lifecycle->mutex);
    if (lifecycle->active_launches == 0) {
      LOG(WARNING) << "NCCL symmetric-memory launch guard released with no "
                      "active launch recorded";
      return;
    }
    --lifecycle->active_launches;
    if (lifecycle->closing && lifecycle->active_launches == 0) {
      lifecycle->cv.notify_all();
    }
  });
}

} // namespace
#endif // USE_ROCM

/* Start of NCCLAllocation implementation */

static StoreExchange storeExchange = StoreExchange("NCCLAllocation");

struct NCCLAllocation {
  // Combined ncclMemAlloc region. Layout (signal pad first):
  //   [0, buffer_offset)                            - signal pad
  //   [buffer_offset, buffer_offset + buffer_size)  - user data buffer
  // buffer_offset equals the signal pad size (already 16-aligned). alloc_base is
  // the ncclMemAlloc base (== signal pad base); alloc() hands back
  // `alloc_base + buffer_offset` (the data buffer).
  void* alloc_base;
  // Size of the user-visible data buffer in bytes, as requested by alloc().
  size_t buffer_size;
  // Byte offset from alloc_base to the start of the user buffer; the signal pad
  // occupies [0, buffer_offset).
  size_t buffer_offset;
  int device_idx;
#ifdef USE_ROCM
  // A cached block can be reused during capture only when its signal pad was
  // zeroed outside capture. Capture-time free marks the block dirty instead of
  // issuing an illegal HIP memset.
  bool signal_pad_clean = true;
  cudaEvent_t signal_pad_zero_event = nullptr;
  // Monotonic insertion counter used by the free-cache byte-budget eviction
  // (oldest block first).
  uint64_t cache_seq = 0;
#endif
  std::mutex mutex;
  // Map of group name to peer alloc info
  ska::flat_hash_map<std::string, c10::intrusive_ptr<NCCLPeerAllocInfo>>
      peer_alloc_infos_;

  NCCLAllocation(
      void* alloc_base,
      size_t buffer_size,
      size_t buffer_offset,
      int device_idx)
      : alloc_base(alloc_base),
        buffer_size(buffer_size),
        buffer_offset(buffer_offset),
        device_idx(device_idx) {}

  ~NCCLAllocation();

#ifdef USE_ROCM
  cudaError_t query_signal_pad_zero() const {
    if (signal_pad_zero_event == nullptr) {
      return cudaSuccess;
    }
    // A query from any thread is prohibited while another thread owns a
    // global-mode capture. Temporarily relax this thread's interaction mode;
    // the event was recorded outside capture, so querying it does not inspect
    // captured work.
    c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard(
        cudaStreamCaptureModeRelaxed);
    const auto query = cudaEventQuery(signal_pad_zero_event);
    if (query == cudaErrorNotReady) {
      // Match the caching allocator's event-query handling: NotReady is an
      // expected readiness result, not an error to leak into later API calls.
      (void)cudaGetLastError();
    }
    return query;
  }

  bool signal_pad_zero_complete() const {
    return query_signal_pad_zero() == cudaSuccess;
  }

  void clear_signal_pad_zero_event() {
    if (signal_pad_zero_event == nullptr) {
      return;
    }
    auto err = cudaEventDestroy(signal_pad_zero_event);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to destroy NCCL symmetric-memory zero event: "
                   << cudaGetErrorString(err);
    }
    signal_pad_zero_event = nullptr;
  }

  bool record_signal_pad_zero(cudaStream_t stream) {
    if (signal_pad_zero_event == nullptr) {
      auto err = cudaEventCreateWithFlags(
          &signal_pad_zero_event, cudaEventDisableTiming);
      if (err != cudaSuccess) {
        LOG(WARNING) << "Failed to create NCCL symmetric-memory zero event: "
                     << cudaGetErrorString(err);
        return false;
      }
    }
    auto err = cudaMemsetAsync(alloc_base, 0, buffer_offset, stream);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to zero NCCL symmetric-memory signal pad: "
                   << cudaGetErrorString(err);
      return false;
    }
    err = cudaEventRecord(signal_pad_zero_event, stream);
    if (err != cudaSuccess) {
      cudaError_t sync_err;
      {
        // A free on this thread may race a global-mode capture owned by
        // another thread. Keep the relaxed window scoped to the synchronous
        // fallback; the async memset and event record above remain unguarded.
        c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard(
            cudaStreamCaptureModeRelaxed);
        sync_err = cudaStreamSynchronize(stream);
      }
      if (sync_err != cudaSuccess) {
        LOG(WARNING) << "Failed to record NCCL symmetric-memory zero event and "
                        "failed to synchronize the cleanup stream: "
                     << cudaGetErrorString(sync_err);
      } else {
        LOG(WARNING) << "Failed to record NCCL symmetric-memory zero event; "
                        "synchronized the cleanup stream instead: "
                     << cudaGetErrorString(err);
      }
      return false;
    }
    signal_pad_clean = true;
    return true;
  }

  bool wait_signal_pad_zero(cudaStream_t stream) {
    if (signal_pad_zero_event == nullptr) {
      return true;
    }
    // This path is not used by a capturing stream, but it may run on a
    // different thread while a global-mode capture is active.
    const auto query = query_signal_pad_zero();
    if (query == cudaSuccess) {
      return true;
    }
    const auto err = cudaStreamWaitEvent(stream, signal_pad_zero_event, 0);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to wait for NCCL symmetric-memory signal pad zero: "
                   << cudaGetErrorString(err);
      return false;
    }
    return true;
  }
#endif
};

namespace {

// Base allocation ptr -> owning NCCL allocation metadata.
// Shared ownership lets rendezvous pin metadata while it releases the global
// allocator lock for NCCL collectives. free()/eviction can remove the mapping
// concurrently, but cannot destroy the allocation until rendezvous rechecks it.
using NCCLAllocMap = ska::flat_hash_map<void*, std::shared_ptr<NCCLAllocation>>;
// (Tensor storage/data ptr, group name) -> cached SymmetricMemory handle.
using NCCLSymmMemMap = ska::flat_hash_map<
    SymmMemKey,
    c10::intrusive_ptr<NCCLSymmetricMemory>,
    SymmMemKeyHash>;
// Base allocation ptr -> cached `(tensor ptr, group)` keys derived from it.
using NCCLSymmMemKeysByAlloc =
    ska::flat_hash_map<void*, ska::flat_hash_set<SymmMemKey, SymmMemKeyHash>>;

bool pointer_in_allocation(void* ptr, const NCCLAllocation& allocation) {
  auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
  // The data buffer starts `buffer_offset` bytes into the allocation (past the
  // signal pad); only data-region pointers belong to this allocation.
  auto buffer_ptr = reinterpret_cast<uintptr_t>(allocation.alloc_base) +
      allocation.buffer_offset;
  return ptr_int >= buffer_ptr && ptr_int < buffer_ptr + allocation.buffer_size;
}

NCCLAllocMap::iterator find_allocation_covering_linear(
    void* ptr,
    NCCLAllocMap& allocations) {
  return std::find_if(
      allocations.begin(),
      allocations.end(),
      [&](const auto& entry) {
        return pointer_in_allocation(ptr, *entry.second);
      });
}

NCCLAllocMap::iterator find_allocation_covering(
    void* ptr,
    NCCLAllocMap& allocations) {
  auto alloc_it = allocations.find(ptr);
  if (alloc_it != allocations.end()) {
    return alloc_it;
  }
  // `ptr` is not an allocation key (a MemPool hands out interior pointers), so
  // scan for the allocation whose [buffer, buffer + size) range covers it. We
  // deliberately do not reconstruct the key from the process-global pad size:
  // get_signal_pad_size() may have changed via set_signal_pad_size() since
  // this allocation was created, whereas the scan uses each allocation's own
  // stored buffer_offset.
  // TODO: this linear std::find_if is O(n) in the number of live allocations.
  // Make it O(log n) by switching NCCLAllocMap to an ordered map and using
  // upper_bound to find the covering allocation.
  return find_allocation_covering_linear(ptr, allocations);
}

} // namespace

// NCCL before 2.29 resolves peer pointers with a device kernel.
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0) && \
    defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT)
#define NCCL_SYMMEM_BUILD_PTR_DEV
// Fill both peer pointer arrays in a single kernel launch. For each peer,
// NCCL returns the window base (== signal pad base); the data buffer pointer
// is derived as `base + buffer_offset`, mirroring the host-side layout.
static __global__ void build_ptr_dev(
  ncclWindow_t  handle,
  size_t  buffer_offset,  // data buffer offset; signal pad occupies [0, buffer_offset)
  void**  buffers,        // out: peer buffer pointers
  void**  signal_pads,    // out: peer signal pad pointers
  int  world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int peer = tid; peer < world_size; peer += stride) {
      void* buf = ncclGetLsaPointer(handle, 0, peer);
      signal_pads[peer] = buf;
      buffers[peer] = buf == nullptr
          ? nullptr
          : static_cast<char*>(buf) + buffer_offset;
  }
}
#endif

class NCCLPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  NCCLPeerAllocInfo(
      NCCLAllocation* allocation,
      std::string group_name)
      : buffer_size_(allocation->buffer_size),
        buffer_offset_(allocation->buffer_offset),
        device_idx_(allocation->device_idx),
        group_name_(std::move(group_name))
  {
    c10::cuda::CUDAGuard guard(device_idx_);
    auto group = resolve_process_group(group_name_);
    rank_ = group->getRank();
    world_size_ = group->getSize();
    // Look up the host ncclComm by group name in NCCLDevCommManager. Any
    // backend that owns a NCCL-compatible communicator (ProcessGroupNCCL, or
    // an external library exposing its ncclComm — torchcomms is one such
    // example) publishes into this registry at comm-init time, so symm_mem
    // doesn't need to know which backend the PG is wrapping.
    auto& mgr = NCCLDevCommManager::get(
        c10::Device(c10::DeviceType::CUDA, device_idx_));
    comm_ = mgr.get_comm(group_name_);
    TORCH_CHECK(
        comm_ != nullptr,
        "NCCL symmetric memory: NCCLDevCommManager returned a null comm for "
        "group '",
        group_name_,
        "'. If you are using ProcessGroups, please make sure its backend has "
        "been eagerly initialized by filling `device_id` in the "
        "`init_process_group` call.");

#ifdef USE_ROCM
    lifecycle_ = get_symm_mem_lifecycle(comm_);
    // Window registration and peer-table construction use the host
    // communicator too. Count the whole construction as an admitted operation
    // so graceful process-group teardown cannot destroy the communicator
    // between registry lookup and the last RCCL call below.
    auto construction_guard = acquire_symm_mem_launch_guard(lifecycle_);
    // RCCL symmetric-memory windows require VMM (cuMem) and window registration
    // (NCCL_CUMEM_ENABLE / NCCL_WIN_ENABLE), both disabled by default in RCCL.
    // RCCL samples these inside ncclCommInitRank -- before symm_mem is ever
    // requested -- so enforce the value recorded at comm-init time rather than
    // re-reading the environment now (it may have changed since init). Fall back
    // to the live environment only when this comm has no snapshot (e.g. a
    // non-PyTorch producer populated the comm registry without recording one).
    const std::optional<bool> precond = rccl_symm_precondition_lookup(comm_);
    const bool precond_ok = precond.has_value()
        ? *precond
        : (c10::utils::check_env("NCCL_CUMEM_ENABLE") == true &&
           c10::utils::check_env("NCCL_WIN_ENABLE") == true);
    TORCH_CHECK(
        precond_ok,
        "RCCL symmetric memory requires NCCL_CUMEM_ENABLE=1 and "
        "NCCL_WIN_ENABLE=1 to be set in the environment before "
        "init_process_group. Set both and re-run.");
#endif

    // Register a single window over the combined signal pad + buffer region.
    // Layout inside the registration (signal pad first):
    //   [0, signal_pad_size)                          - signal pad
    //   [buffer_offset_, buffer_offset_ + buffer)     - user data buffer
    // The single registration sidesteps NCCL's window-alignment requirement
    // for the data sub-region: only the base pointer (returned by
    // ncclMemAlloc, already granularity-aligned) is registered.
    const size_t aligned_buffer_size = at::round_up(buffer_size_, 16UL);
#ifdef USE_ROCM
    // RCCL additionally requires the registered window size to be a multiple of
    // NCCL_WIN_REQUIRED_ALIGNMENT. Upstream NCCL only constrains the base
    // offset, so keep the size round-up ROCm-only to avoid changing CUDA
    // allocation and registration sizes.
    const size_t total_size = at::round_up(
        buffer_offset_ + aligned_buffer_size,
        static_cast<size_t>(NCCL_WIN_REQUIRED_ALIGNMENT));
#else
    const size_t total_size = buffer_offset_ + aligned_buffer_size;
#endif
    C10D_NCCL_CHECK(
      ncclCommWindowRegister(comm_, allocation->alloc_base, total_size, &combined_win_, NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register segment with ptr ",
          allocation->alloc_base,
          ", size ",
          total_size,
          " on rank ",
          rank_));

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
    // (Host comm is already published into NCCLDevCommManager by the
    // owning backend at comm-init time. The earlier mgr.get_comm() call
    // above relied on that. No re-register here.)

    // Starting from NCCL 2.28, we can get peer pointers.
    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    buffers_.resize(world_size_);
    signal_pads_.resize(world_size_);

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
    // No host-side peer-pointer API is available, so a kernel resolves both
    // peer arrays at once via ncclGetLsaPointer and copies them to host.
    int threads = std::min(128, world_size_);
    auto stream = at::cuda::getCurrentCUDAStream();
    build_ptr_dev<<<1, threads, 0, stream>>>(
        combined_win_, buffer_offset_, buffers_dev_, signal_pads_dev_, world_size_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    C10_CUDA_CHECK(cudaMemcpy(
      buffers_.data(),  // dst (host)
      buffers_dev_,  // src (device)
      arr_size,
      cudaMemcpyDeviceToHost));
    C10_CUDA_CHECK(cudaMemcpy(
      signal_pads_.data(),  // dst (host)
      signal_pads_dev_,  // src (device)
      arr_size,
      cudaMemcpyDeviceToHost));
#else
  // Starting from NCCL 2.29, we can use host-side APIs to get peer pointers.
  // ncclGetPeerDevicePointer returns each peer's window base, which is the
  // signal pad base (the signal pad is at the front of the window).
  for (int i = 0; i < world_size_; i++) {
    // If peer is not accessible within LSA domain, `ncclGetPeerDevicePointer`
    // returns nullptr.
    C10D_NCCL_CHECK(
      ncclGetPeerDevicePointer(combined_win_, 0, i, &signal_pads_[i]),
      "ncclGetPeerDevicePointer failed");
  }
  // Derive each peer's data buffer pointer from its window base; all ranks
  // share the same buffer_offset_ so we don't need to ask NCCL separately.
  for (int i = 0; i < world_size_; i++) {
    buffers_[i] = signal_pads_[i] == nullptr
        ? nullptr
        : static_cast<char*>(signal_pads_[i]) + buffer_offset_;
  }
#ifdef USE_ROCM
  // Synchronous copies implicitly use the legacy stream and are forbidden
  // during HIP graph capture. The source vectors live with this peer info.
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      buffers_dev_,
      buffers_.data(),
      arr_size,
      cudaMemcpyHostToDevice,
      stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      signal_pads_dev_,
      signal_pads_.data(),
      arr_size,
      cudaMemcpyHostToDevice,
      stream));
#else
  C10_CUDA_CHECK(cudaMemcpy(
      buffers_dev_, buffers_.data(), arr_size, cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(
      signal_pads_dev_,
      signal_pads_.data(),
      arr_size,
      cudaMemcpyHostToDevice));
#endif

  // Starting from NCCL 2.29, we can use `ncclGetLsaMultimemDevicePointer`
  // to get multicast address.
  void* mc_addr = nullptr;
  // Skip CHECK on purpose to improve fault tolerance since some machine's
  // Fabric Manager may be in bad NVLink Sharp state.
  // Pass buffer_offset_ as the window offset so the returned multicast pointer
  // already points at the data buffer (past the signal pad); no manual add.
  if (ncclGetLsaMultimemDevicePointer(
          combined_win_, buffer_offset_, &mc_addr) == ncclSuccess &&
      mc_addr != nullptr) {
    mc_addr_ = mc_addr;
  }
#endif // NCCL_VERSION_CODE < NCCL_VERSION(2, 29, 0)
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  }

  // Exact copy is not needed / supported
  NCCLPeerAllocInfo(const NCCLPeerAllocInfo& other) = delete;
  NCCLPeerAllocInfo& operator=(const NCCLPeerAllocInfo& other) = delete;
  NCCLPeerAllocInfo(NCCLPeerAllocInfo&& other) = delete;
  NCCLPeerAllocInfo& operator=(NCCLPeerAllocInfo&& other) = delete;

  ~NCCLPeerAllocInfo() {
    try {
      release_window();
      if (!is_finalizing()) {
        c10::cuda::CUDAGuard guard(device_idx_);
        if (buffers_dev_ != nullptr) {
          c10::cuda::CUDACachingAllocator::raw_delete(buffers_dev_);
        }
        if (signal_pads_dev_ != nullptr) {
          c10::cuda::CUDACachingAllocator::raw_delete(signal_pads_dev_);
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to release NCCL peer allocation info: "
                   << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to release NCCL peer allocation info";
    }
  }

  // The host communicator this info's window was registered on.
  ncclComm_t host_comm() const {
    return comm_;
  }

  int device_idx() const {
    return device_idx_;
  }

#ifdef USE_ROCM
  bool can_reclaim_device_tables_after_sync() const {
    if (!lifecycle_) {
      return false;
    }
    std::lock_guard<std::mutex> lock(lifecycle_->mutex);
    return lifecycle_->closing && lifecycle_->active_launches == 0 &&
        lifecycle_->reclaimable_after_sync;
  }
#endif

  bool is_live() const {
#ifdef USE_ROCM
    if (comm_invalidated_.load(std::memory_order_acquire)) {
      return false;
    }
    if (lifecycle_) {
      std::lock_guard<std::mutex> lock(lifecycle_->mutex);
      return !lifecycle_->closing;
    }
    return true;
#else
    return true;
#endif
  }

  void check_live() const {
    TORCH_CHECK(
        is_live(),
        "This symmetric-memory handle is bound to a destroyed communicator or "
        "freed backing allocation. "
        "Rendezvous the tensor again with a live process group.");
  }

  // Deregister while the communicator is live and make the operation
  // idempotent for external SymmetricMemory handles that retain this object.
  void release_window() {
    if (combined_win_ == nullptr || is_finalizing()) {
      return;
    }
#ifdef USE_ROCM
    // A closing communicator still owns a live window that must be deregistered.
    // Only invalidation means teardown took ownership and this object must stop.
    if (comm_invalidated_.load(std::memory_order_acquire)) {
      return;
    }
    // A normal eviction may race process-group shutdown. If teardown has
    // already closed admission, the communicator owns this window now and
    // invalidation below will prevent a later deregistration. Otherwise the
    // lease makes graceful teardown wait for this host API call to finish.
    NCCLSymmetricMemoryLaunchGuard release_guard;
    try {
      release_guard = acquire_symm_mem_launch_guard(lifecycle_);
    } catch (const c10::Error&) {
      return;
    }
#endif
    auto window = combined_win_;
    combined_win_ = nullptr;
    try {
      c10::cuda::CUDAGuard guard(device_idx_);
      auto res = ncclCommWindowDeregister(comm_, window);
      if (res != ncclSuccess) {
        LOG(WARNING) << "ncclCommWindowDeregister failed: "
                     << ncclGetErrorString(res);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to deregister NCCL symmetric-memory window: "
                   << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to deregister NCCL symmetric-memory window";
    }
  }

#ifdef USE_ROCM
  // The communicator owns and reclaims the window at teardown. Keep the stale
  // value only so this object's destructor can distinguish invalidation from a
  // live window that still needs explicit deregistration; every public use is
  // rejected by check_live().
  void invalidate() {
    comm_invalidated_.store(true, std::memory_order_release);
  }
#endif

 private:
  size_t buffer_size_;
  // Byte offset from the allocation base to the start of the user buffer; the
  // signal pad occupies [0, buffer_offset_).
  size_t buffer_offset_;
  int device_idx_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_{nullptr};
  void** signal_pads_dev_{nullptr};
  std::string group_name_;
  // Single NCCL window covering both the user data buffer and the signal pad.
  ncclWindow_t combined_win_{nullptr};
  // Multicast address (data buffer base within the multicast mapping)
  void* mc_addr_{nullptr};
  ncclComm_t comm_{nullptr};
#ifdef USE_ROCM
  std::atomic<bool> comm_invalidated_{false};
  std::shared_ptr<SymmMemCommLifecycle> lifecycle_;
#endif
  friend class NCCLSymmetricMemory;
};

NCCLAllocation::~NCCLAllocation() {
  // Avoid calling CUDA functions after driver shutting down.
  if (is_finalizing()) {
    return;
  }
  try {
    c10::cuda::CUDAGuard guard(device_idx);
#ifdef USE_ROCM
    // Windows must be released before their backing allocation. Explicitly
    // invalidate retained handles after deregistration and before ncclMemFree.
    for (auto& [_, pai] : peer_alloc_infos_) {
      if (pai) {
        pai->release_window();
        pai->invalidate();
      }
    }
    peer_alloc_infos_.clear();
    clear_signal_pad_zero_event();
#endif
    // Single free for the combined buffer + signal pad region.
    ncclResult_t res = ncclMemFree(alloc_base);
    if (res != ncclSuccess) {
      LOG(WARNING) << "ncclMemFree failed in NCCLAllocation dtor: "
                   << ncclGetErrorString(res);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to free NCCL symmetric-memory allocation: "
                 << e.what();
  } catch (...) {
    LOG(WARNING) << "Failed to free NCCL symmetric-memory allocation";
  }
}

NCCLSymmetricMemory::NCCLSymmetricMemory(
    c10::intrusive_ptr<NCCLPeerAllocInfo> pai,
    size_t offset)
    : pai_(std::move(pai)),
      offset_(offset),
      rank_(pai_->rank_),
      world_size_(pai_->world_size_),
      device_idx_(pai_->device_idx_) {
  TORCH_INTERNAL_ASSERT(offset_ < pai_->buffer_size_, "offset out of range");
}

bool NCCLSymmetricMemory::is_live_for(ncclComm_t comm) const {
  return pai_->is_live() && pai_->host_comm() == comm;
}

NCCLSymmetricMemoryLaunchGuard NCCLSymmetricMemory::acquire_launch_guard()
    const {
#ifdef USE_ROCM
  pai_->check_live();
  return acquire_symm_mem_launch_guard(pai_->lifecycle_);
#else
  return NCCLSymmetricMemoryLaunchGuard();
#endif
}

std::vector<void*> NCCLSymmetricMemory::get_buffer_ptrs() {
#ifdef USE_ROCM
  pai_->check_live();
#endif
  return pai_->buffers_;
}

std::vector<void*> NCCLSymmetricMemory::get_signal_pad_ptrs() {
#ifdef USE_ROCM
  pai_->check_live();
#endif
  return pai_->signal_pads_;
}

#ifdef USE_ROCM
// On ROCm the dev-side pointer tables are only populated at RCCL >= 2.29.7 (LSA
// peer-pointer API). Ops that launch kernels reading these tables would
// otherwise dereference null/garbage on intermediate RCCL versions where only
// the host-side window registration exists. CUDA populates the tables whenever
// symmetric memory is available, so it keeps the plain accessor (unchanged).
static constexpr const char* kPeerPtrsUnavailable =
    "device-side peer pointer tables were not populated; symmetric-memory "
    "peer access requires every peer to be reachable over the LSA domain and "
    "RCCL >= 2.30.4";
#endif

void** NCCLSymmetricMemory::get_buffer_ptrs_dev() {
#ifdef USE_ROCM
  pai_->check_live();
  TORCH_CHECK(pai_->buffers_dev_ != nullptr, kPeerPtrsUnavailable);
#endif
  return pai_->buffers_dev_;
}

void** NCCLSymmetricMemory::get_signal_pad_ptrs_dev() {
#ifdef USE_ROCM
  pai_->check_live();
  TORCH_CHECK(pai_->signal_pads_dev_ != nullptr, kPeerPtrsUnavailable);
#endif
  return pai_->signal_pads_dev_;
}

size_t NCCLSymmetricMemory::get_buffer_size() {
  return pai_->buffer_size_;
}

bool NCCLSymmetricMemory::has_multicast_support() {
#ifdef USE_ROCM
  pai_->check_live();
#endif
  return pai_->mc_addr_ != nullptr;
}

void* NCCLSymmetricMemory::get_multicast_ptr() {
#ifdef USE_ROCM
  pai_->check_live();
#endif
  if (!has_multicast_support()) {
    return nullptr;
  }
  return static_cast<char*>(pai_->mc_addr_) + offset_;
}

void NCCLSymmetricMemory::barrier(int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#ifdef USE_ROCM
  auto launch_guard = acquire_launch_guard();
#endif
  TORCH_CHECK(
      pai_->signal_pads_dev_ != nullptr,
      "NCCLSymmetricMemory::barrier requires peer signal pad pointers, which "
      "are only populated when peers are accessible over the symmetric-memory "
      "(LSA/NVLink) domain.");
  check_channel(channel, world_size_, get_signal_pad_size());
  c10::cuda::CUDAGuard device_guard(device_idx_);
  barrier_kernel<<<
      1,
      std::max(at::cuda::warp_size(), world_size_),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      channel,
      rank_,
      world_size_,
      timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(false, "NYI");
#endif
}

void NCCLSymmetricMemory::put_signal(int dst_rank, int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_ONE_SIDED_API
#ifdef USE_ROCM
  auto launch_guard = acquire_launch_guard();
#endif
  check_rank(dst_rank, world_size_);
  TORCH_CHECK(channel == 0, "channel must be 0 (sigIdx is reserved for future use)");

  c10::cuda::CUDAGuard guard(device_idx_);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto& manager = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
  ncclComm_t comm = manager.get_comm(pai_->group_name_);

  // use ncclSignal for pure signaling without data transfer
  C10D_NCCL_CHECK(
      ncclSignal(
          dst_rank,
          channel,
          0,
          0,
          comm,
          stream),
      c10::str("ncclSignal failed for dst_rank=", dst_rank, ", channel=", channel));
#else
  TORCH_CHECK(false, "NYI");
#endif
}

void NCCLSymmetricMemory::wait_signal(int src_rank, int channel, size_t timeout_ms) {
#ifdef NCCL_HAS_ONE_SIDED_API
#ifdef USE_ROCM
  auto launch_guard = acquire_launch_guard();
#endif
  check_rank(src_rank, world_size_);
  TORCH_CHECK(channel == 0, "channel must be 0 (sigIdx is reserved for future use)");

  c10::cuda::CUDAGuard guard(device_idx_);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto& manager = NCCLDevCommManager::get(c10::Device(c10::DeviceType::CUDA, device_idx_));
  ncclComm_t comm = manager.get_comm(pai_->group_name_);

  // create signal descriptor for waiting - populate all fields
  ncclWaitSignalDesc_t signalDesc;
  signalDesc.opCnt = 1;
  signalDesc.peer = src_rank;
  signalDesc.sigIdx = channel;
  signalDesc.ctx = 0;

  C10D_NCCL_CHECK(
      ncclWaitSignal(
          1,
          &signalDesc,
          comm,
          stream),
      c10::str("ncclWaitSignal failed for src_rank=", src_rank, ", channel=", channel));
#else
  TORCH_CHECK(false, "NYI");
#endif
}

int NCCLSymmetricMemory::get_rank() {
  return rank_;
}

int NCCLSymmetricMemory::get_world_size() {
  return world_size_;
}

c10::Device NCCLSymmetricMemory::get_device() {
  return c10::Device(c10::DeviceType::CUDA, device_idx_);
}

ncclWindow_t NCCLSymmetricMemory::get_window() {
#ifdef USE_ROCM
  pai_->check_live();
#endif
  return pai_->combined_win_;
}

size_t NCCLSymmetricMemory::get_offset() {
  return offset_;
}

size_t NCCLSymmetricMemory::get_window_offset() {
#ifdef USE_ROCM
  pai_->check_live();
#endif
  // The NCCL window starts at the signal pad; this handle's data lives
  // buffer_offset_ bytes further in, plus its own offset within the buffer.
  return pai_->buffer_offset_ + offset_;
}

#ifdef NCCL_HAS_HOST_CFT
// Both CFT queries only succeed once NCCL has created the communicator's
// logical endpoints. With `hostCftMode` enabled that happens during the first
// `ncclCommWindowRegister`, i.e. during rendezvous; otherwise the caller would
// have had to build a CFT-enabled `ncclDevComm` first.
static constexpr const char* kHostCftHint =
    "Host-side CFT requires CFT-capable hardware and a process group whose "
    "communicator was created with `host_cft_mode` enabled "
    "(ProcessGroupNCCL.NCCLConfig.host_cft_mode).";
#endif // NCCL_HAS_HOST_CFT

NCCLCftHandle NCCLSymmetricMemory::get_peer_cft_handle(int peer) {
#ifdef NCCL_HAS_HOST_CFT
  TORCH_CHECK(
      peer >= 0 && peer < world_size_,
      "NCCLSymmetricMemory::get_peer_cft_handle: invalid peer ",
      peer);
  c10::cuda::CUDAGuard guard(device_idx_);
  ncclCftLeId le_id = 0;
  size_t le_offset = 0;
  C10D_NCCL_CHECK(
      ncclGetPeerDeviceLeInfo(
          pai_->combined_win_, get_window_offset(), peer, &le_id, &le_offset),
      c10::str(
          "ncclGetPeerDeviceLeInfo failed for peer ", peer, ". ", kHostCftHint));
  return NCCLCftHandle{le_id, le_offset};
#else
  TORCH_CHECK(
      false, "NCCL host-side CFT is not supported. Requires NCCL >= 2.31.2");
#endif
}

NCCLCftHandle NCCLSymmetricMemory::get_multimem_cft_handle() {
#ifdef NCCL_HAS_HOST_CFT
  // Unlike the unicast query, this one may still have to bind the multicast
  // team (and barrier over the group) if the endpoint wasn't created eagerly.
  c10::cuda::CUDAGuard guard(device_idx_);
  ncclCftLeId le_id = 0;
  size_t le_offset = 0;
  C10D_NCCL_CHECK(
      ncclGetMultimemDeviceLeInfo(
          pai_->combined_win_, get_window_offset(), &le_id, &le_offset),
      c10::str("ncclGetMultimemDeviceLeInfo failed. ", kHostCftHint));
  return NCCLCftHandle{le_id, le_offset};
#else
  TORCH_CHECK(
      false, "NCCL host-side CFT is not supported. Requires NCCL >= 2.31.2");
#endif
}

std::string NCCLSymmetricMemory::get_group_name() {
  return pai_->group_name_;
}

class NCCLSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  // Allocates a symmetric-memory region laid out as [signal pad | data buffer]:
  // the signal pad occupies [0, buffer_offset) and the user data buffer starts
  // at buffer_offset. Returns the data buffer pointer (alloc_base +
  // buffer_offset), NOT the allocation base -- the signal pad stays hidden in
  // front, and free()/rendezvous() key off this returned data pointer.
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(
        group_name == std::nullopt,
        "NCCLSymmetricMemoryAllocator::alloc "
        "must not be called with a group_name");

    c10::cuda::CUDAGuard guard(device_idx);
    // Allocate signal pad + buffer together in a single ncclMemAlloc call.
    // Layout: signal pad in [0, buffer_offset), data buffer after it.
    // buffer_offset is the signal pad size rounded up to signal_pad_alignment,
    // so the data buffer is aligned; the data size is rounded up as well. A
    // single window is registered over the whole region at rendezvous time, so
    // only the base pointer (already granularity-aligned by ncclMemAlloc) needs
    // to satisfy NCCL's window-alignment requirement.
    const size_t buffer_offset =
        at::round_up(get_signal_pad_size(), signal_pad_alignment);
    const size_t aligned_buffer_size = at::round_up(size, 16UL);
#ifdef USE_ROCM
    // RCCL requires the registered window size (== this allocation size) to be
    // a multiple of NCCL_WIN_REQUIRED_ALIGNMENT; upstream NCCL does not, so the
    // round-up stays ROCm-only to keep CUDA allocation sizes unchanged.
    const size_t total_size = at::round_up(
        buffer_offset + aligned_buffer_size,
        static_cast<size_t>(NCCL_WIN_REQUIRED_ALIGNMENT));
#else
    const size_t total_size = buffer_offset + aligned_buffer_size;
#endif

#ifdef USE_ROCM
    const bool in_capture =
        c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
        c10::cuda::CaptureStatus::None;
    std::shared_ptr<NCCLAllocation> cached_alloc;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      FreeCacheKey cache_key{size, buffer_offset, device_idx};
      auto cache_it = free_cache_.find(cache_key);
      if (cache_it != free_cache_.end()) {
        auto& blocks = cache_it->second;
        auto block_it = blocks.end();
        // A rendezvous temporarily owns another shared_ptr while it performs
        // NCCL collectives without the allocator lock. Reusing that same object
        // here would let free -> same-address alloc form an ABA cycle and pass
        // rendezvous's final identity check for a different tensor.
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
          // signal_pad_clean means a zero and its event were enqueued outside
          // capture. The guarded query is both capture-conformant and needed:
          // frees on other threads can enqueue a same-size block after
          // torch.cuda.graph's pre-capture synchronize.
          if (it->use_count() == 1 &&
              (!in_capture ||
               ((*it)->signal_pad_clean &&
                (*it)->signal_pad_zero_complete()))) {
            // Preserve the normal LIFO reuse policy among eligible blocks.
            block_it = it;
          }
        }
        if (block_it != blocks.end()) {
          cached_alloc = std::move(*block_it);
          free_cache_bytes_[device_idx] -=
              cached_alloc->buffer_offset + cached_alloc->buffer_size;
          blocks.erase(block_it);
          if (blocks.empty()) {
            free_cache_.erase(cache_it);
          }
        }
      }
    }
    if (cached_alloc) {
      c10::cuda::CUDAGuard guard(device_idx);
      auto stream = at::cuda::getCurrentCUDAStream().stream();
      if (in_capture) {
        // The guarded eligibility query above proved the pre-capture zero
        // complete. Do not query again or wait on its event here: a wait on an
        // event recorded outside capture would add an external dependency to
        // the graph.
        TORCH_INTERNAL_ASSERT(cached_alloc->signal_pad_clean);
      } else if (cached_alloc->signal_pad_clean) {
        TORCH_CHECK(
            cached_alloc->wait_signal_pad_zero(stream),
            "Failed to order NCCL symmetric-memory signal pad cleanup before reuse.");
      } else {
        TORCH_CHECK(
            cached_alloc->record_signal_pad_zero(stream),
            "Failed to zero NCCL symmetric-memory signal pad before reuse.");
      }
      TORCH_INTERNAL_ASSERT(cached_alloc->buffer_size == size);
      void* buffer_ptr = static_cast<char*>(cached_alloc->alloc_base) +
          cached_alloc->buffer_offset;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [_, inserted] =
            allocations_.emplace(buffer_ptr, std::move(cached_alloc));
        TORCH_INTERNAL_ASSERT(inserted);
      }
      return buffer_ptr;
    }
    TORCH_CHECK(
        !in_capture,
        "NCCLSymmetricMemoryAllocator::alloc called during HIP graph capture "
        "without a clean cached block for size=",
        size,
        ". Call symm_mem.empty() and rendezvous() with this exact size "
        "outside capture, then synchronize before capturing the graph.");
#endif // USE_ROCM

    void* alloc_base;
    C10D_NCCL_CHECK(ncclMemAlloc(&alloc_base, total_size), "ncclMemAlloc");
    // ncclMemAlloc does not zero memory. Zero the signal pad (the first
    // buffer_offset bytes) so the CAS-based barrier() protocol starts from a
    // known all-zero state on first use.
#ifdef USE_ROCM
    C10_CUDA_CHECK(cudaMemsetAsync(
        alloc_base, 0, buffer_offset, at::cuda::getCurrentCUDAStream().stream()));
#else
    C10_CUDA_CHECK(cudaMemset(alloc_base, 0, buffer_offset));
#endif
    // Hand back the data buffer pointer, not alloc_base; the signal pad stays
    // hidden in front. Returning the data ptr is safe for free(): the whole
    // block is owned by the NCCLAllocation keyed below, which ncclMemFree's
    // alloc_base in its destructor, so free() only needs the data ptr to drop
    // the allocation entry.
    void* buffer_ptr = static_cast<char*>(alloc_base) + buffer_offset;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Key by the data pointer we return (that's what `free()` receives).
      allocations_.emplace(
          buffer_ptr,
          std::make_shared<NCCLAllocation>(
              alloc_base, size, buffer_offset, device_idx));
    }
    return buffer_ptr;
  }

  void free(void* ptr) override {
#ifdef USE_ROCM
    // Callers must quiesce kernels using this allocation before dropping the
    // tensor. PyTorch cannot track custom kernels that use exposed peer
    // pointers, so free() only orders allocator-owned signal-pad cleanup before
    // a later reuse.
    std::shared_ptr<NCCLAllocation> nccl_alloc;
    int device_idx = -1;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto alloc_it = allocations_.find(ptr);
      if (alloc_it == allocations_.end()) {
        return;
      }
      nccl_alloc = alloc_it->second;
      device_idx = nccl_alloc->device_idx;
      // Drop the cached SymmetricMemory handles for this block so a
      // post-free rendezvous fails instead of returning a handle to memory
      // nobody owns. The peer alloc infos themselves stay alive inside
      // nccl_alloc->peer_alloc_infos_ (they move with the block into the
      // free cache), so reusing the block -- including during graph capture,
      // where re-registering a window is illegal -- still finds them without
      // any new NCCL calls.
      erase_symm_mem_handles(ptr);
      allocations_.erase(alloc_it);
    }

    bool in_capture = true;
    try {
      c10::cuda::CUDAGuard guard(device_idx);
      in_capture = c10::cuda::currentStreamCaptureStatusMayInitCtx() !=
          c10::cuda::CaptureStatus::None;
      if (!in_capture) {
        // Best-effort cleanup for graph-capture reuse. The zero event orders
        // later reuse and keeps eviction from freeing an in-flight memset.
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        nccl_alloc->signal_pad_clean = nccl_alloc->record_signal_pad_zero(stream);
      } else {
        nccl_alloc->signal_pad_clean = false;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to query/clean NCCL symmetric-memory free block: "
                   << e.what();
      nccl_alloc->signal_pad_clean = false;
    } catch (...) {
      LOG(WARNING) << "Failed to query/clean NCCL symmetric-memory free block";
      nccl_alloc->signal_pad_clean = false;
    }

    std::vector<std::shared_ptr<NCCLAllocation>> eviction_victims;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      FreeCacheKey cache_key{
          nccl_alloc->buffer_size,
          nccl_alloc->buffer_offset,
          nccl_alloc->device_idx};
      nccl_alloc->cache_seq = next_cache_seq_++;
      free_cache_bytes_[device_idx] +=
          nccl_alloc->buffer_offset + nccl_alloc->buffer_size;
      free_cache_[cache_key].push_back(std::move(nccl_alloc));
      // Eviction ncclMemFree's blocks and deregisters their windows, which is
      // illegal mid-capture; let the budget go transiently over instead.
      if (!in_capture) {
        evict_free_cache_to(
            device_idx, kFreeCacheByteBudget, eviction_victims);
      }
    }
    // Destroying victims may deregister RCCL windows and ncclMemFree their
    // backing blocks. RCCL operations are rank-local, so ranks may evict at
    // different times as long as their own kernels have quiesced.
    eviction_victims.clear();
#else
    std::lock_guard<std::mutex> lock(mutex_);
    auto alloc_it = allocations_.find(ptr);
    if (alloc_it == allocations_.end()) {
      return;
    }
    auto cache_keys_it = symm_mem_keys_by_alloc_.find(ptr);
    if (cache_keys_it != symm_mem_keys_by_alloc_.end()) {
      for (const auto& key : cache_keys_it->second) {
        symm_mems_.erase(key);
      }
      symm_mem_keys_by_alloc_.erase(cache_keys_it);
    }
    allocations_.erase(alloc_it);
#endif
  };

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
      TORCH_CHECK(
          false, ptr, " is not allocated with NCCLSymmetricMemoryAllocator");
    }
    return it->second->buffer_size;
  };

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    TORCH_CHECK(group_name.has_value(), "group_name must be provided");
    std::shared_ptr<NCCLAllocation> allocation;
    // The covering allocation's map key is buffer_ptr (the data buffer base
    // alloc() returned, == alloc_base + buffer_offset); captured here so we
    // don't recompute it below.
    void* buffer_ptr_key = nullptr;
    SymmMemKey key{ptr, *group_name};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Find the allocation covering the ptr under the allocator lock.
      // Pin the allocation while the allocator lock is released. A concurrent
      // free/cache eviction may remove it from allocations_, but destruction is
      // deferred until the identity check below rejects this rendezvous.
      auto alloc_it = find_allocation_covering(ptr, allocations_);
      TORCH_CHECK(
          alloc_it != allocations_.end(),
          "Pointer not within any SymmetricMemory allocation, "
          "is the tensor allocated from SymmetricMemory?");
      auto it = symm_mems_.find(key);
      if (it != symm_mems_.end()) {
#ifdef USE_ROCM
        auto* live_comm =
            NCCLDevCommManager::get(
                c10::Device(
                    c10::DeviceType::CUDA, alloc_it->second->device_idx))
                .get_comm(*group_name);
        TORCH_CHECK(
            it->second->is_live_for(live_comm),
            "The cached symmetric-memory handle belongs to a replaced "
            "communicator. Rendezvous the tensor again.");
#endif
        return it->second;
      }
      allocation = alloc_it->second;
      buffer_ptr_key = alloc_it->first;
    }

    c10::intrusive_ptr<NCCLPeerAllocInfo> pai;
    {
      // Serialize peer-info creation, but release this lock before reacquiring
      // the allocator lock below. Communicator invalidation takes them in the
      // opposite phase: allocator first, then allocation.
      std::lock_guard<std::mutex> alloc_lock(allocation->mutex);
      auto& cached_pai = allocation->peer_alloc_infos_[*group_name];
#ifdef USE_ROCM
      // A same-name successor communicator needs its own window and peer
      // tables. Invalidate the predecessor PAI, but keep the ncclMemAlloc block
      // itself reusable because allocations are not communicator-owned.
      if (cached_pai) {
        auto* live_comm =
            NCCLDevCommManager::get(
                c10::Device(c10::DeviceType::CUDA, allocation->device_idx))
                .get_comm(*group_name);
        if (!cached_pai->is_live() || cached_pai->host_comm() != live_comm) {
          cached_pai->invalidate();
          cached_pai.reset();
        }
      }
#endif
      if (!cached_pai) {
#ifdef USE_ROCM
        TORCH_CHECK(
            c10::cuda::currentStreamCaptureStatusMayInitCtx() ==
                c10::cuda::CaptureStatus::None,
            "NCCLSymmetricMemoryAllocator::rendezvous would register an NCCL "
            "window during HIP graph capture. Call rendezvous() for this block "
            "and group outside capture before capturing the graph.");
#endif
        cached_pai =
            c10::make_intrusive<NCCLPeerAllocInfo>(allocation.get(), *group_name);
      }
      pai = cached_pai;
    }
    // Offset is relative to the data buffer base (past the signal pad).
    size_t offset = reinterpret_cast<uintptr_t>(ptr) -
        reinterpret_cast<uintptr_t>(buffer_ptr_key);
    // Create the SymmetricMemory handle.
    auto symm_mem = c10::make_intrusive<NCCLSymmetricMemory>(pai, offset);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto alloc_it = allocations_.find(buffer_ptr_key);
      TORCH_CHECK(
          alloc_it != allocations_.end() &&
              alloc_it->second.get() == allocation.get(),
          "The symmetric-memory allocation was freed during rendezvous.");
#ifdef USE_ROCM
      auto* live_comm =
          NCCLDevCommManager::get(
              c10::Device(c10::DeviceType::CUDA, allocation->device_idx))
              .get_comm(*group_name);
      TORCH_CHECK(
          pai->is_live() && pai->host_comm() == live_comm,
          "The process-group communicator changed during rendezvous. "
          "Rendezvous the tensor again.");
#endif
      // Insert the SymmetricMemory handle into the map (cache), keyed by the
      // (Tensor storage ptr, group name) pair.
      auto [it, inserted] = symm_mems_.emplace(key, symm_mem);
      if (!inserted) {
        // This condition should rarely happen, only when another thread happens
        // to be concurrently rendezvousing with the same allocation for the
        // same group.  For safety, we return the existing SymmetricMemory
        // handle and discard the new one.
        return it->second;
      }
      // There is no more use of `key`; we can move it into the per-allocation
      // key set to avoid an extra copy. Key by the data pointer (the value
      // returned by alloc()), matching the lookup done in free().
      symm_mem_keys_by_alloc_[buffer_ptr_key].insert(std::move(key));
    }
    return symm_mem;
  }

  bool has_multicast_support(int device_idx) override {
    return device_has_multicast_support(device_idx);
  }

  bool has_allocation(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    return find_allocation_covering(ptr, allocations_) != allocations_.end();
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CUDA;
  }

  std::string name() override {
    return "NCCL";
  }

#ifdef USE_ROCM
  void invalidate_for_comm(
      int device_idx,
      const std::string& group_name,
      ncclComm_t comm,
      bool reclaim_device_tables) {
    // Keep invalidated PAIs alive until both allocator and per-allocation locks
    // have been released. Abort/undrained teardown retires them instead of
    // freeing device-pointer tables that a kernel may still read.
    std::vector<c10::intrusive_ptr<NCCLPeerAllocInfo>> invalidated_pais;
    std::lock_guard<std::mutex> lock(mutex_);

    auto invalidate = [&](NCCLAllocation* allocation) {
      if (allocation->device_idx != device_idx) {
        return false;
      }
      std::lock_guard<std::mutex> alloc_lock(allocation->mutex);
      auto it = allocation->peer_alloc_infos_.find(group_name);
      if (it == allocation->peer_alloc_infos_.end() || !it->second ||
          it->second->host_comm() != comm) {
        return false;
      }
      // The communicator teardown owns the old window. Invalidate externally
      // retained handles and remove only this communicator's PAI; the backing
      // ncclMemAlloc block and PAIs for unrelated groups remain reusable.
      auto pai = std::move(it->second);
      pai->invalidate();
      allocation->peer_alloc_infos_.erase(it);
      invalidated_pais.push_back(std::move(pai));
      return true;
    };

    for (auto& [ptr, allocation] : allocations_) {
      if (invalidate(allocation.get())) {
        erase_symm_mem_handles_for_group(ptr, group_name);
      }
    }

    for (auto& [_, blocks] : free_cache_) {
      for (auto& block : blocks) {
        invalidate(block.get());
      }
    }
    if (!reclaim_device_tables) {
      retired_pais_.insert(
          retired_pais_.end(),
          std::make_move_iterator(invalidated_pais.begin()),
          std::make_move_iterator(invalidated_pais.end()));
    }
  }

  void reclaim_retired_for_device(int device_idx) {
    std::vector<c10::intrusive_ptr<NCCLPeerAllocInfo>> reclaimed_pais;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = retired_pais_.begin(); it != retired_pais_.end();) {
        if ((*it)->device_idx() == device_idx &&
            (*it)->can_reclaim_device_tables_after_sync()) {
          reclaimed_pais.push_back(std::move(*it));
          it = retired_pais_.erase(it);
        } else {
          ++it;
        }
      }
    }
    reclaimed_pais.clear();
  }

#endif // USE_ROCM

 private:
  void erase_symm_mem_handles(void* ptr) {
    auto keys_it = symm_mem_keys_by_alloc_.find(ptr);
    if (keys_it == symm_mem_keys_by_alloc_.end()) {
      return;
    }
    for (const auto& key : keys_it->second) {
      symm_mems_.erase(key);
    }
    symm_mem_keys_by_alloc_.erase(keys_it);
  }

#ifdef USE_ROCM
  void erase_symm_mem_handles_for_group(
      void* ptr,
      const std::string& group_name) {
    auto keys_it = symm_mem_keys_by_alloc_.find(ptr);
    if (keys_it == symm_mem_keys_by_alloc_.end()) {
      return;
    }
    auto& keys = keys_it->second;
    for (auto it = keys.begin(); it != keys.end();) {
      if (it->second == group_name) {
        symm_mems_.erase(*it);
        it = keys.erase(it);
      } else {
        ++it;
      }
    }
    if (keys.empty()) {
      symm_mem_keys_by_alloc_.erase(keys_it);
    }
  }

  // Exact user size and signal-pad layout are part of the key. Different
  // requests can round to the same NCCL window size but must not share cached
  // rendezvous metadata, and blocks cached before a set_signal_pad_size()
  // change must never be handed out against the new layout.
  struct FreeCacheKey {
    size_t buffer_size;
    size_t buffer_offset;
    int device_idx;
    bool operator==(const FreeCacheKey& other) const {
      return buffer_size == other.buffer_size &&
          buffer_offset == other.buffer_offset && device_idx == other.device_idx;
    }
  };
  struct FreeCacheKeyHash {
    size_t operator()(const FreeCacheKey& key) const {
      return c10::get_hash(key.buffer_size, key.buffer_offset, key.device_idx);
    }
  };
  std::unordered_map<
      FreeCacheKey,
      std::vector<std::shared_ptr<NCCLAllocation>>,
      FreeCacheKeyHash>
      free_cache_;

  // Per-device bytes sitting in free_cache_. Capture status is device-local, so
  // eviction must never destroy blocks from a device whose capture state was
  // not queried.
  ska::flat_hash_map<int, size_t> free_cache_bytes_;
  uint64_t next_cache_seq_ = 0;
  std::vector<c10::intrusive_ptr<NCCLPeerAllocInfo>> retired_pais_;

  // Drop oldest blocks for device_idx until that device is within budget.
  // Must not be called during capture on device_idx.
  void evict_free_cache_to(
      int device_idx,
      size_t budget,
      std::vector<std::shared_ptr<NCCLAllocation>>& victims) {
    auto& device_bytes = free_cache_bytes_[device_idx];
    while (device_bytes > budget) {
      NCCLAllocation* oldest = nullptr;
      FreeCacheKey oldest_key{0, 0, 0};
      size_t oldest_idx = 0;
      for (auto& [key, blocks] : free_cache_) {
        if (key.device_idx != device_idx) {
          continue;
        }
        for (size_t i = 0; i < blocks.size(); i++) {
          if (!blocks[i]->signal_pad_zero_complete()) {
            continue;
          }
          if (oldest == nullptr ||
              blocks[i]->cache_seq < oldest->cache_seq) {
            oldest = blocks[i].get();
            oldest_key = key;
            oldest_idx = i;
          }
        }
      }
      if (oldest == nullptr) {
        return;
      }
      auto& blocks = free_cache_[oldest_key];
      device_bytes -= blocks[oldest_idx]->buffer_offset +
          blocks[oldest_idx]->buffer_size;
      victims.push_back(std::move(blocks[oldest_idx]));
      blocks.erase(blocks.begin() + static_cast<long>(oldest_idx));
      if (blocks.empty()) {
        free_cache_.erase(oldest_key);
      }
    }
  }
#endif // USE_ROCM

  std::mutex mutex_;
  NCCLAllocMap allocations_;
  NCCLSymmMemMap symm_mems_;
  NCCLSymmMemKeysByAlloc symm_mem_keys_by_alloc_;
};

// The process-wide NCCL symmetric-memory allocator singleton. Kept as a raw
// pointer so communicator teardown can invalidate its allocations; the
// intrusive_ptr held by the allocator registry keeps it alive for the process
// lifetime.
static NCCLSymmetricMemoryAllocator* nccl_symm_allocator_ = nullptr;

struct RegisterNCCLSymmetricMemoryAllocator {
    RegisterNCCLSymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<NCCLSymmetricMemoryAllocator>();
    nccl_symm_allocator_ = allocator.get();
    // Query backend used for CUDA tensor
    if (getSymmMemBackendCUDA() == "NCCL") {
      // Direct set (static registration)
      register_allocator(
          c10::DeviceType::CUDA,
          allocator);
    } else {
      // Register availability in case `set_backend` is called dynamically
      register_availability("NCCL", allocator);
    }
  }
};

static RegisterNCCLSymmetricMemoryAllocator register_allocator_;

bool begin_symm_mem_teardown_for_comm(
    void* comm,
    std::chrono::milliseconds timeout,
    bool* drained) {
#ifdef USE_ROCM
  if (drained != nullptr) {
    *drained = false;
  }
  auto lifecycle = find_symm_mem_lifecycle(static_cast<ncclComm_t>(comm));
  if (!lifecycle) {
    return false;
  }
  std::unique_lock<std::mutex> lock(lifecycle->mutex);
  lifecycle->closing = true;
  if (!lifecycle->used) {
    return false;
  }
  const bool did_drain = lifecycle->cv.wait_for(
      lock, timeout, [&]() { return lifecycle->active_launches == 0; });
  if (!did_drain) {
    lifecycle->reclaimable_after_sync = false;
  }
  if (drained != nullptr) {
    *drained = did_drain;
  }
  return true;
#else
  (void)comm;
  (void)timeout;
  if (drained != nullptr) {
    *drained = false;
  }
  return false;
#endif
}

bool close_symm_mem_for_comm(void* comm) {
#ifdef USE_ROCM
  auto lifecycle = find_symm_mem_lifecycle(static_cast<ncclComm_t>(comm));
  if (!lifecycle) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lifecycle->mutex);
  lifecycle->closing = true;
  lifecycle->reclaimable_after_sync = false;
  return lifecycle->used;
#else
  (void)comm;
  return false;
#endif
}

void invalidate_symm_mem_for_comm(
    const c10::Device& device,
    const std::string& group_name,
    void* comm,
    bool reclaim_device_tables) {
#ifdef USE_ROCM
  if (nccl_symm_allocator_ != nullptr) {
    nccl_symm_allocator_->invalidate_for_comm(
        static_cast<int>(device.index()),
        group_name,
        static_cast<ncclComm_t>(comm),
        reclaim_device_tables);
  }
  forget_symm_mem_lifecycle(static_cast<ncclComm_t>(comm));
#else
  (void)device;
  (void)group_name;
  (void)comm;
  (void)reclaim_device_tables;
#endif
}

void reclaim_retired_symm_mem_for_device(const c10::Device& device) {
#ifdef USE_ROCM
  if (nccl_symm_allocator_ != nullptr) {
    nccl_symm_allocator_->reclaim_retired_for_device(
        static_cast<int>(device.index()));
  }
#else
  (void)device;
#endif
}

} // namespace symmetric_memory
} // namespace c10d
#endif // NCCL_HAS_SYMMEM_SUPPORT
