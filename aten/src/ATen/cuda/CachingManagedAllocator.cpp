#include <ATen/cuda/CachingManagedAllocator.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/irange.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace at {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace cuda {
namespace CachingManagedAllocator {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

namespace {

using stream_set = std::unordered_set<cuda::CUDAStream>;

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB

typedef std::bitset<static_cast<size_t>(StatType::NUM_TYPES)> StatTypes;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT(
      stat.current >= 0,
      "Negative tracked stat in CUDA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

void update_stat_array(StatArray& stat_array,
    uint64_t amount,
    const StatTypes& stat_types) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      update_stat(stat_array[stat_type], amount);
    }
  }
}

struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);

struct BlockPool {
  BlockPool(
      Comparison comparator,
      bool small,
      PrivatePool* private_pool = nullptr)
      : blocks(comparator), is_small(small), owner_PrivatePool(private_pool) {}
  std::set<Block*, Comparison> blocks;
  const bool is_small;
  PrivatePool* owner_PrivatePool;
};

struct Block {
  int device; // gpu
  cudaStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  BlockPool* pool; // owning memory pool
  void* ptr; // memory address
  bool allocated; // in-use flag
  Block* prev; // prev block if split from a larger allocation
  Block* next; // next block if split from a larger allocation
  int event_count; // number of outstanding CUDA events

  Block(
      int device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(cudaSuccess) {}

  int device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types;
  cudaError_t err;
};

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

cudaError_t cudaMallocMaybeCapturing(void** p, size_t size) {
    return cudaMallocManaged(p, size);
}

} // namespace 

class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

 private:
  static std::once_flag s_flag;
  static CachingAllocatorConfig* s_instance;
  static CachingAllocatorConfig& instance() {
    std::call_once(s_flag, &CachingAllocatorConfig::init);
    return *s_instance;
  }
  static void init() {
    s_instance = new CachingAllocatorConfig();
    s_instance->parseArgs();
  }

  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()) {}
  size_t m_max_split_size;

  void parseArgs() {
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != NULL) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          /* Maximum split size in MB.  Limited to large size blocks */
          if (kv[0].compare("max_split_size_mb") == 0) {
            size_t val2 = stoi(kv[1]);
            TORCH_CHECK(
                val2 > kLargeBuffer / (1024 * 1024),
                "CachingAllocator option max_split_size_mb too small, must be > ",
                kLargeBuffer / (1024 * 1024),
                "");
            val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
            val2 = std::min(
                val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
            m_max_split_size = val2 * 1024 * 1024;
          } else {
            TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
          }
        }
      }
    }
  }
};
CachingAllocatorConfig* CachingAllocatorConfig::s_instance;
std::once_flag CachingAllocatorConfig::s_flag;

class ManagedAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // lock around calls to cudaFree (to prevent deadlocks with NCCL)
  mutable std::mutex cuda_free_mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  std::unordered_set<Block*> active_blocks;

  //TODO:  This is going to be redundant with active_blocks.  Need to merge
  //the two going forward.
  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // captures_underway tracks if a capture might be underway on any stream.
  // Most of the time it's zero, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.

  // See free() for this thing's purpose
  std::vector<Block*> needs_events_deferred_until_no_capture;
  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, Block*>> cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

  // Maps a capturing stream to its assigned private pool,
  // in case we want multiple captures to share the same pool
  std::unordered_map<CaptureId_t, MempoolId_t> capture_to_pool_map;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
  ManagedAllocator()
      : large_blocks(BlockComparator, /*is_small=*/false),
        small_blocks(BlockComparator, /*is_small=*/true) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
  }

  std::mutex* getCudaFreeMutex() const {
    return &cuda_free_mutex;
  }

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  cudaError_t recordEvent(void* ptr, at::cuda::CUDAStream stream)
  {

    return cudaSuccess;
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.
void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    Block* block = malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
  }

  Block* malloc(int device, size_t size, cudaStream_t stream) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    process_events();
    size = round_size(size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params))
        // Attempt allocate
        || alloc_block(params, false)
        // Free enough available cached blocks to satisfy alloc and retry alloc.
        ||
        (release_available_cached_blocks(params) && alloc_block(params, false))
        // Free all non-split cached blocks and retry alloc.
        || (release_cached_blocks() && alloc_block(params, true));

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

      size_t device_free;
      size_t device_total;
      C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      stats.num_ooms += 1;

    }

    TORCH_INTERNAL_ASSERT(
        params.err == cudaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (should_split(block, size)) {
      remaining = block;

      block = new Block(device, stream, size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool.blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(
            stats.inactive_split_bytes, -block->size, params.stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        update_stat_array(
            stats.inactive_split_bytes, remaining->size, params.stat_types);
        update_stat_array(stats.inactive_split, 1, params.stat_types);
      }
    } else if (already_split) {
      // An already-split block is becoming active
      update_stat_array(
          stats.inactive_split_bytes, -block->size, params.stat_types);
      update_stat_array(stats.inactive_split, -1, params.stat_types);
    }

    block->allocated = true;
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    update_stat_array(stats.allocation, 1, params.stat_types);
    update_stat_array(stats.allocated_bytes, block->size, params.stat_types);
    update_stat_array(stats.active, 1, params.stat_types);
    update_stat_array(stats.active_bytes, block->size, params.stat_types);
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        block->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));

    add_allocated_block(block);
    return block;
  }

  void free(void* ptr) {

    std::lock_guard<std::recursive_mutex> lock(mutex);

    // We aren't sure if a free'd block will be used by the same device
    // or even the same stream, synchronize for now just in case
    cudaDeviceSynchronize();
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
        true;
    update_stat_array(stats.allocation, -1, {stat_types});
    update_stat_array(stats.allocated_bytes, -block->size, {stat_types});
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
        insert_events(block);
    } else {
      free_block(block);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
      if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
  }

  /** Retrieves info (total size + largest block) of the memory cache **/
  void cacheInfo(size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes;
      cudaMemGetInfo(
          largest, // Use free memory as an optimistic initial guess of *largest
          &tmp_bytes);
    }
    cache_info_aux(large_blocks, total, largest);
    cache_info_aux(small_blocks, total, largest);
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.is_large = (!head_block->pool->is_small);

      const Block* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
        }

        block = block->next;
      }
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    return result;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    size_t original_block_size = block->size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    update_stat_array(
        stats.inactive_split, net_change_inactive_split_blocks, stat_types);
    update_stat_array(
        stats.inactive_split_bytes, net_change_inactive_split_size, stat_types);
    update_stat_array(stats.active, -1, stat_types);
    update_stat_array(stats.active_bytes, -original_block_size, stat_types);
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size, cudaStream_t stream) {

    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatType get_stat_type_for_pool(const BlockPool& pool) {
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  bool alloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      p.err = cudaMallocMaybeCapturing(&ptr, size);
      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear
          // CUDA's
          //   internal error state.
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take
          //   over to throw a helpful exception. The user can choose to catch
          //   the exception, free some stuff in their script, and attempt their
          //   allocation again. In this case, we can also forgive and clear
          //   CUDA's internal error state.
          cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    update_stat_array(stats.segment, 1, p.stat_types);
    update_stat_array(stats.reserved_bytes, size, p.stat_types);
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;
    Block key = p.search_key;
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks() {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);

    return true;
  }

  void release_block(Block* block) {
    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
    update_stat_array(stats.segment, -1, stat_types);
    update_stat_array(stats.reserved_bytes, -block->size, stat_types);
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);

    pool->blocks.erase(block);
    delete block;
  }

  void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  cudaEvent_t create_event_internal() {
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  void free_event_internal(cudaEvent_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }

  void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.

    for (auto& e : cuda_events) {
      cudaEvent_t event = e.first;
      Block* block = e.second;

      C10_CUDA_CHECK(cudaEventSynchronize(event));
      free_event_internal(event);

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block) {
    int prev_device;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_CUDA_CHECK(cudaSetDevice(stream.device_index()));

      cudaEvent_t event = create_event_internal();
      C10_CUDA_CHECK(cudaEventRecord(event, stream.stream()));

      block->event_count++;
      cuda_events.emplace_back(event, block);
    }

    C10_CUDA_CHECK(cudaSetDevice(prev_device));
  }

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(needs_events_deferred_until_no_capture.size() > 0)) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        insert_events(block);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events() {
    insert_events_deferred_until_no_capture();

    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!cuda_events.empty()) {
      auto& e = cuda_events.front();
      cudaEvent_t event = e.first;
      Block* block = e.second;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) {
        // ignore and clear the error if not ready
        cudaGetLastError();
        break;
      } else if (err != cudaSuccess) {
        C10_CUDA_CHECK(err);
      }

      free_event_internal(event);

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      cuda_events.pop_front();
    }
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* total, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

};

static ManagedAllocator allocator;

 void setMemoryFraction(double fraction, int device) {
   allocator.setMemoryFraction(fraction);
 }

 void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
   allocator.cacheInfo(cachedAndFree, largestBlock);
 }

 void* getBaseAllocation(void* ptr, size_t* size) {
   return allocator.getBaseAllocation(ptr, size);
 }

 void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
   allocator.recordStream(ptr, stream);
 }

std::mutex* getFreeMutex() {
  return allocator.getCudaFreeMutex();
}

DeviceStats getDeviceStats() {
   return allocator.getStats();
}

void resetAccumulatedStats(int device) {
   allocator.resetAccumulatedStats();
}

void resetPeakStats(int device) {
  allocator.resetPeakStats();
}

std::vector<SegmentInfo> snapshot() {
  return allocator.snapshot();
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  allocator.malloc(
      &r, device, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  allocator.malloc(&r, device, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  allocator.free(ptr);
}

cudaError_t recordEvent(void *ptr, at::cuda::CUDAStream stream)
{
  return allocator.recordEvent(ptr, stream);
}

void emptyCache()
{
  allocator.emptyCache();
}

static void CachingManagedDeleter(void* ptr) {
  allocator.free(ptr);
}
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  C10_CUDA_CHECK(cudaFree(ptr));
}

struct CachingManagedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* ptr = nullptr;
    if (forceUncachedAllocator()) {
      // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
      // if someone tries to use forceUncachedAllocator while capturing.
      C10_CUDA_CHECK(cudaMallocManaged(&ptr, size));
      return {ptr, ptr, &uncached_delete, Device(DeviceType::CUDA, device)};
    }
    if (size != 0) {
      ptr = allocator.malloc(device, size, cuda::getCurrentCUDAStream(device))->ptr;
      //C10_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
    }
    return {ptr, ptr, &CachingManagedDeleter, Device(DeviceType::CUDA, device)};
  }
  at::DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &CachingManagedDeleter;
    }
  }
};

struct CachingManagedAllocatorCpu final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void* ptr = nullptr;
    if (forceUncachedAllocator()) {
      // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
      // if someone tries to use forceUncachedAllocator while capturing.
      C10_CUDA_CHECK(cudaMallocManaged(&ptr, size));
      return {ptr, ptr, &uncached_delete, Device(DeviceType::CPU)};
    }
    if (size != 0) {
      int device;
      C10_CUDA_CHECK(cudaGetDevice(&device));
      int hint_device = -1; //cudaCpuDeviceId
      ptr = allocator.malloc(hint_device, size, cuda::getCurrentCUDAStream(device))->ptr;

      //C10_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, hint_device));
      //C10_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device));
    }
    return {ptr, ptr, &CachingManagedDeleter, Device(DeviceType::CPU)};
  }
  at::DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &CachingManagedDeleter;
    }
  }
};

static CachingManagedAllocator caching_managed_allocator;

static CachingManagedAllocatorCpu caching_managed_allocator_cpu;

at::Allocator* get(DeviceType device_type) {
  if (device_type == DeviceType::CUDA)
    return &caching_managed_allocator;
  else
    return &caching_managed_allocator_cpu;
}

} // namespace CachingManagedAllocator

}} // namespace at::cuda
