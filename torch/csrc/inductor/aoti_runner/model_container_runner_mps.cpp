#if defined(__APPLE__)
#include <torch/csrc/inductor/aoti_runner/model_container_runner_mps.h>

namespace torch::inductor {

AOTIModelContainerRunnerMps::AOTIModelContainerRunnerMps(
    const std::string& model_so_path,
    size_t num_models,
    bool run_single_threaded)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          "mps",
          "",
          run_single_threaded) {}

AOTIModelContainerRunnerMps::~AOTIModelContainerRunnerMps() = default;

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_mps(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
<<<<<<< HEAD
  TORCH_CHECK(
      device_str == "mps", "Incorrect device passed to aoti_runner_mps");
=======
  if (device_str != "mps") {
    throw std::runtime_error("Incorrect device passed to aoti_runner_mps");
  }
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  return std::make_unique<AOTIModelContainerRunnerMps>(
      model_so_path, num_models, run_single_threaded);
}
} // namespace

static RegisterAOTIModelRunner register_mps_runner(
    "mps",
    &create_aoti_runner_mps);

} // namespace torch::inductor
#endif
