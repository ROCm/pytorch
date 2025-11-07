#pragma once

#include <ATen/Config.h>

#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>

namespace at::native::onednn {

TORCH_XPU_API dnnl::memory make_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr);

// Keep non-static and non-inline
bool set_onednn_verbose(int level);

// GpuEngineManager singleton
struct TORCH_XPU_API GpuEngineManager {
  static GpuEngineManager& Instance(); // Singleton

<<<<<<< HEAD
  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    TORCH_INTERNAL_ASSERT(device.index() < c10::xpu::device_count());
    return *engine_pool[device.index()];
=======
  dnnl::engine& get_engine(
      DeviceIndex device_index = c10::xpu::current_device()) {
    c10::xpu::check_device_index(device_index);
    return *engine_pool[device_index];
  }

  dnnl::engine& get_engine(const Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == kXPU);
    return get_engine(device.index());
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;
  GpuEngineManager(GpuEngineManager&&) = default;
  GpuEngineManager& operator=(GpuEngineManager&&) = default;

 protected:
  GpuEngineManager();
  ~GpuEngineManager() = default;

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct TORCH_XPU_API GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

<<<<<<< HEAD
  dnnl::stream get_stream() {
    auto stream = c10::xpu::getCurrentXPUStream();
    auto priority = stream.priority();
    auto device_index = stream.device_index();
=======
  dnnl::stream& get_stream(
      DeviceIndex device_index = c10::xpu::current_device()) {
    auto stream = c10::xpu::getCurrentXPUStream(device_index);
    auto priority = stream.priority();
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    if (stream_pool[device_index][priority].find(stream) ==
        stream_pool[device_index][priority].end()) {
      stream_pool[device_index][priority][stream] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
<<<<<<< HEAD
              GpuEngineManager::Instance().get_engine(
                  {c10::kXPU, device_index}),
=======
              GpuEngineManager::Instance().get_engine(device_index),
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
              stream.queue()));
    }
    return *stream_pool[device_index][priority][stream];
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;
  GpuStreamManager(GpuStreamManager&&) = default;
  GpuStreamManager& operator=(GpuStreamManager&&) = default;

 protected:
  GpuStreamManager() {
<<<<<<< HEAD
    c10::DeviceIndex device_count = c10::xpu::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
=======
    c10::DeviceIndex device_count = c10::xpu::device_count_ensure_non_zero();
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    stream_pool.resize(device_count);
  }
  ~GpuStreamManager() = default;

 private:
  using stream_hash_map =
      ska::flat_hash_map<c10::xpu::XPUStream, std::shared_ptr<dnnl::stream>>;
  std::vector<
      std::array<stream_hash_map, c10::xpu::max_compile_time_stream_priorities>>
      stream_pool;
};

} // namespace at::native::onednn
