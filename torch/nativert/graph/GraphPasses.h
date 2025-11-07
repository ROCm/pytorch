#pragma once

#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

<<<<<<< HEAD
void selectScalarOverload(Graph* graph);

std::string selectScalarOverloadName(const Node& node);
=======
void selectScalarOverload(torch::nativert::Graph* graph);

std::string selectScalarOverloadName(const torch::nativert::Node& node);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

} // namespace torch::nativert
