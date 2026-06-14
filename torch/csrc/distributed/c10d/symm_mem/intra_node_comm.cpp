#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#if defined(USE_ROCM)
#include <dlfcn.h>
#endif

namespace c10d::intra_node_comm {

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
// Forces detectedTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without NVLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};
static int intraNodeCommIdx = 0;

/**
 * Query the nvlink connection among devices.
 */
static NvlMesh getNvlMesh(const std::vector<int>& rankToDeviceIdx) {
#if !defined(USE_ROCM)
  auto connectivity = detect_dma_connectivity(c10::DeviceType::CUDA, "nvlink");
  NvlMesh nvlMesh = {};
  for (size_t srcRank = 0; srcRank < kMaxDevices; ++srcRank) {
    for (size_t dstRank = 0; dstRank < kMaxDevices; ++dstRank) {
      if (srcRank < rankToDeviceIdx.size() &&
          dstRank < rankToDeviceIdx.size()) {
        nvlMesh[srcRank][dstRank] =
            connectivity
                ->matrix[rankToDeviceIdx[srcRank]][rankToDeviceIdx[dstRank]];
      }
    }
  }
  return nvlMesh;
#else
  // All AMDSMI symbols are resolved at runtime via dlsym so that
  // libtorch_hip.so carries no undefined rsmi_*/amdsmi_* symbols and
  // no link-time NEEDED dependency on libamd_smi.so.
  using amdsmi_handle = void*;
  using amdsmi_init_fn = int (*)(uint64_t);
  using amdsmi_shut_down_fn = int (*)();
  using amdsmi_get_socket_handles_fn = int (*)(uint32_t*, amdsmi_handle*);
  using amdsmi_get_processor_handles_fn =
      int (*)(amdsmi_handle, uint32_t*, amdsmi_handle*);
  using amdsmi_is_P2P_accessible_fn =
      int (*)(amdsmi_handle, amdsmi_handle, bool*);

  constexpr uint64_t AMDSMI_INIT_AMD_GPUS = (1 << 1);

  auto load = [](const char* name) -> void* {
    void* sym = dlsym(RTLD_DEFAULT, name);
    if (!sym) {
      LOG(ERROR) << "IntraNodeComm: dlsym(" << name << ") failed: "
                 << dlerror();
    }
    return sym;
  };

  auto fn_init = reinterpret_cast<amdsmi_init_fn>(load("amdsmi_init"));
  auto fn_shut = reinterpret_cast<amdsmi_shut_down_fn>(
      load("amdsmi_shut_down"));
  auto fn_sockets = reinterpret_cast<amdsmi_get_socket_handles_fn>(
      load("amdsmi_get_socket_handles"));
  auto fn_procs = reinterpret_cast<amdsmi_get_processor_handles_fn>(
      load("amdsmi_get_processor_handles"));
  auto fn_p2p = reinterpret_cast<amdsmi_is_P2P_accessible_fn>(
      load("amdsmi_is_P2P_accessible"));

  if (!fn_init || !fn_shut || !fn_sockets || !fn_procs || !fn_p2p) {
    return {};
  }

  if (fn_init(AMDSMI_INIT_AMD_GPUS) != 0) {
    LOG(ERROR) << "IntraNodeComm: amdsmi_init failed";
    return {};
  }

  // Collect GPU processor handles.
  uint32_t socket_count = 0;
  fn_sockets(&socket_count, nullptr);
  std::vector<amdsmi_handle> sockets(socket_count);
  fn_sockets(&socket_count, sockets.data());

  std::vector<amdsmi_handle> gpuHandles;
  for (uint32_t s = 0; s < socket_count; ++s) {
    uint32_t dev_count = 0;
    fn_procs(sockets[s], &dev_count, nullptr);
    std::vector<amdsmi_handle> devs(dev_count);
    fn_procs(sockets[s], &dev_count, devs.data());
    gpuHandles.insert(gpuHandles.end(), devs.begin(), devs.end());
  }

  NvlMesh nvlMesh = {};
  const auto worldSize = rankToDeviceIdx.size();
  // For each device, loop over devices connected to it
  for (size_t idx = 0; idx < worldSize; ++idx) {
    for (size_t link = 0; link < kMaxDevices; ++link) {
      if (idx == link)
        continue;
      if (idx >= gpuHandles.size() || link >= gpuHandles.size())
        continue;

      bool conn = false;
      auto ret = fn_p2p(gpuHandles[idx], gpuHandles[link], &conn);
      if (ret != 0) {
        LOG(ERROR)
            << "IntraNodeComm: getNvlMesh: amdsmi_is_P2P_accessible"
               " returned error ret="
            << ret;
        fn_shut();
        return {};
      }

      if (conn) {
        nvlMesh[idx][link] += 1;
      }
    }
  }
  fn_shut();
  return nvlMesh;
#endif
}

/**
 * Detect topology given a NvlMesh.
 */
static Topology detectTopology(const NvlMesh nvlMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }
  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (nvlMesh[i][j] == 0 || nvlMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  LOG(INFO) << "IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
}

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      bufferSize_(bufferSize.has_value() ? *bufferSize : kDefaultBufferSize) {}

IntraNodeComm::~IntraNodeComm() {
  if (!isInitialized_) {
    return;
  }
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  allocator->free(symmetricMemoryPtr_);
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

/**
 * Use c10d::Store to perform allgather on a trivially copyable type.
 */
template <typename T>
static std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << '-' << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
  if (!isIntraNodeCommSupported() || worldSize_ < 2 ||
      worldSize_ > kMaxDevices) {
    return false;
  }

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  deviceIdx_ = at::cuda::current_device();

  // Exchange hostname and device bus ID
  struct DevInfo {
    // NOLINTNEXTLINE
    char hostname[HOST_NAME_MAX + 1];
    int deviceIdx;
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  devInfo.deviceIdx = deviceIdx_;

  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<int> rankToDeviceIdx;
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING) << "Aborting IntraNodeComm::rendezvous because some "
                      "participants are not on the same host ("
                   << info.hostname << ", " << devInfo.hostname << ')';
      return false;
    }
    rankToDeviceIdx.emplace_back(info.deviceIdx);
  }

  {
    std::unordered_set uniqueDeviceIdxs(
        rankToDeviceIdx.begin(), rankToDeviceIdx.end());
    if (uniqueDeviceIdxs.size() != worldSize_) {
      LOG(WARNING)
          << "Skipping IntraNodeComm::rendezvous() because participants have "
             "overlapping devices. To resolve this, call torch.cuda.set_device() "
             "before init_process_group().";
      return false;
    }
  }

  // Query nvlink connection
  auto nvlMesh = getNvlMesh(rankToDeviceIdx);

  // Detect topology
  topology_ = detectTopology(nvlMesh, worldSize_);
  if (topology_ != Topology::FULLY_CONNECTED) {
    return false;
  }

  auto groupName = "IntraNodeComm" + std::to_string(intraNodeCommIdx++);
  set_group_info(
      groupName, static_cast<int>(rank_), static_cast<int>(worldSize_), store_);
  auto allocator = get_allocator(c10::DeviceType::CUDA);
  symmetricMemoryPtr_ = allocator->alloc(bufferSize_, deviceIdx_, groupName);
  symmetricMemory_ = allocator->rendezvous(symmetricMemoryPtr_, std::nullopt);
  isInitialized_ = true;
  return true;
}

} // namespace c10d::intra_node_comm
