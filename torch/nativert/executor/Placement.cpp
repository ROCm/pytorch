#include <torch/nativert/executor/Placement.h>

#include <fmt/ostream.h>
#include <ostream>

namespace torch::nativert {

std::ostream& operator<<(std::ostream& os, const Placement& placement) {
  std::vector<std::pair<std::string, c10::Device>> sorted_keys;
  sorted_keys.reserve(placement.deviceMap_.size());
  for (const auto& pair : placement.deviceMap_) {
    sorted_keys.emplace_back(pair.first.str(), pair.first);
  }
  std::sort(
      sorted_keys.begin(), sorted_keys.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

  bool first = true;
  for (const auto& pair : sorted_keys) {
    if (!first) {
      fmt::print(os, ",");
    }
    first = false;
    const auto& key = pair.second;
    const auto& value = placement.deviceMap_.at(key);
    fmt::print(os, "{}|{}", pair.first, value.str());
  }
  if (placement.defaultDevice_.has_value()) {
    fmt::print(os, "{}|{}", first ? "" : ",", placement.defaultDevice_->str());
  }
  return os;
}

<<<<<<< HEAD
namespace {
void assertCudaDeviceHasIndex(const c10::Device& device) {
  if (device.is_cuda()) {
    TORCH_CHECK(
        device.has_index(), "CUDA device in placement must have an index");
  }
}
} // namespace

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
Placement::Placement(std::optional<c10::Device> defaultDevice)
    : Placement({}, defaultDevice) {}

Placement::Placement(
    const std::unordered_map<c10::Device, c10::Device>& deviceMap,
    std::optional<c10::Device> defaultDevice) {
  for (const auto& [srcDevice, dstDevice] : deviceMap) {
<<<<<<< HEAD
    assertCudaDeviceHasIndex(srcDevice);
    assertCudaDeviceHasIndex(dstDevice);

    deviceMap_.try_emplace(srcDevice, dstDevice);
  }

  if (defaultDevice.has_value()) {
    assertCudaDeviceHasIndex(defaultDevice.value());
    defaultDevice_ = defaultDevice.value();
=======
    deviceMap_.try_emplace(
        normalizeDevice(srcDevice), normalizeDevice(dstDevice));
  }
  if (defaultDevice.has_value()) {
    defaultDevice_ = normalizeDevice(defaultDevice.value());
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  }
}

c10::Device Placement::getMappedDevice(const c10::Device& srcDevice) const {
<<<<<<< HEAD
  auto it = deviceMap_.find(srcDevice);
=======
  auto it = deviceMap_.find(normalizeDevice(srcDevice));
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  if (it != deviceMap_.end()) {
    return it->second;
  }
  if (defaultDevice_.has_value()) {
    return defaultDevice_.value();
  }
  return srcDevice;
}

} // namespace torch::nativert
