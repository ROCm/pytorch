#pragma once

// Shim of cuDNN's `<cudnn_frontend.h>` for ROCm/hipDNN builds. Forwards
// `cudnn_frontend` symbols to `hipdnn_frontend` with cuDNN-style API shims
// layered on top (Graph::check_support(handle), HeurMode_t::A, etc.), and
// also forwards cuDNN-side `cudnnHandle_t`/`getCudnnHandle()`/
// `AT_CUDNN_FRONTEND_CHECK` to hipDNN equivalents — so non-hipified cuDNN
// call sites compile unchanged on HIP.

// TODO: drop this define once hipDNN exposes SDPA unconditionally and
// pytorch's LoadHIP.cmake propagates hipdnn_frontend's
// INTERFACE_COMPILE_DEFINITIONS.
#define HIPDNN_ENABLE_SDPA

#include <ATen/hipdnn/Exceptions.h>
#include <ATen/hipdnn/Handle.h>
#include <ATen/hipdnn/hipdnn-wrapper.h>
#include <hipdnn_frontend.hpp>

namespace at::native::hipdnn_compat {

using namespace hipdnn_frontend;

namespace graph {
using namespace hipdnn_frontend::graph;

class Graph : public hipdnn_frontend::graph::Graph {
 public:
  // cuDNN's check_support / build_plans take a handle; hipDNN's don't (the
  // handle is bound at execute time). Add overloads that ignore the handle
  // and forward to the no-arg APIs.
  using hipdnn_frontend::graph::Graph::check_support;
  using hipdnn_frontend::graph::Graph::build_plans;
  auto check_support(hipdnnHandle_t /*handle*/) { return check_support(); }
  auto build_plans(hipdnnHandle_t /*handle*/) { return build_plans(); }

  // cuDNN exposes a per-uid query via an out-parameter. hipDNN only offers
  // a one-shot {uid -> shared_ptr<Tensor_attributes>} map; wrap it.
  hipdnn_frontend::error_t query_tensor_attributes_of_uid(
      int64_t uid,
      hipdnn_frontend::graph::Tensor_attributes& attrs) const {
    auto graph_tensors = getTensorsByUid();
    auto it = graph_tensors.find(uid);
    if (it == graph_tensors.end()) {
      return {hipdnn_frontend::error_code_t::ATTRIBUTE_NOT_SET,
              "tensor uid not in graph"};
    }
    attrs = *it->second;
    return {hipdnn_frontend::error_code_t::OK, ""};
  }
};

} // namespace graph

// Map cuDNN's HeurMode_t::A (recommended heuristic) to FALLBACK on hipDNN.
struct HeurMode_t {
  static constexpr auto A = hipdnn_frontend::HeurMode_t::FALLBACK;
};

} // namespace at::native::hipdnn_compat

namespace cudnn_frontend = at::native::hipdnn_compat;
