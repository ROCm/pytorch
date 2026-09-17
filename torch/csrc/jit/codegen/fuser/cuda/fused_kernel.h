#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch::jit::fuser::cuda {

// query codegen output arch and target
TORCH_CUDA_CU_API void codegenOutputQuery(
    const hipDeviceProp_t* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass);

// A class holding metadata for an actual CUDA function.
// Note: CUDA functions are per device.
struct TORCH_CUDA_CU_API FusedKernelCUDA
    : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCUDA(
      at::DeviceIndex device,
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  ~FusedKernelCUDA() override;

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override;

  at::Backend backend() const override {
    return at::Backend::CUDA;
  }

 private:
  static constexpr auto kBlockSize = 128;

  // Note: per device to store device properties and compute launch heuristics
  //  Acquiring these values at launch time would be too slow
  at::DeviceIndex device_;
  int maxBlocks_{};
  hipDeviceProp_t* prop_{};
  std::vector<char> ptx_;
  hipModule_t module_{};
  hipFunction_t function_{};
};

} // namespace torch::jit::fuser::cuda
