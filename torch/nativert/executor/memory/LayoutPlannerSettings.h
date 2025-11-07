#pragma once

#include <chrono>

namespace torch::nativert {

enum class LayoutPlannerAlgorithmType {
  Bump,
  GreedyBySize,
<<<<<<< HEAD
  DisjointStorageGroups,
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
};

class LayoutManagerSettings {
 public:
  LayoutManagerSettings() = default;

  bool deallocateBetweenRequests() const {
    return deallocateBetweenRequests_;
  }

  LayoutManagerSettings& setDeallocateBetweenRequests(
      bool deallocateBetweenRequests) {
    deallocateBetweenRequests_ = deallocateBetweenRequests;
    return *this;
  }

 private:
  friend class LayoutManager;
  bool deallocateBetweenRequests_{true};
};

class LayoutPlannerSettings {
 public:
  LayoutPlannerSettings() = default;

  bool enabled() const {
    return enabled_;
  }

  LayoutPlannerAlgorithmType algorithmType() const {
    return layoutPlannerAlgorithmType_;
  }

  std::chrono::seconds planningInterval() const {
    return planningInterval_;
  }

  const LayoutManagerSettings& layoutManagerSettings() const {
    return layoutManagerSettings_;
  }

  LayoutPlannerSettings& setEnabled(bool enabled) {
    enabled_ = enabled;
    return *this;
  }

  LayoutPlannerSettings& setAlgorithmType(
      LayoutPlannerAlgorithmType layoutPlannerAlgorithmType) {
    layoutPlannerAlgorithmType_ = layoutPlannerAlgorithmType;
    return *this;
  }

  LayoutPlannerSettings& setPlanningInterval(
      std::chrono::seconds planningInterval) {
    planningInterval_ = planningInterval;
    return *this;
  }

  LayoutPlannerSettings& setLayoutManagerSettings(
      LayoutManagerSettings layoutManagerSettings) {
    layoutManagerSettings_ = layoutManagerSettings;
    return *this;
  }

 private:
  friend class LayoutPlanner;
  bool enabled_{false};
  LayoutPlannerAlgorithmType layoutPlannerAlgorithmType_{
      LayoutPlannerAlgorithmType::Bump};
  std::chrono::seconds planningInterval_{5};
  LayoutManagerSettings layoutManagerSettings_;
};

} // namespace torch::nativert
