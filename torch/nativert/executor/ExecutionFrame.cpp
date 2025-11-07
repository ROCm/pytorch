#include <c10/util/Enumerate.h>
#include <c10/util/Logging.h>

#include <torch/nativert/executor/ExecutionFrame.h>
<<<<<<< HEAD
=======
#include <torch/nativert/executor/ExecutionPlanner.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

namespace torch::nativert {

ExecutionFrame::ExecutionFrame(const Graph& graph)
    : graph_(graph),
      allValues_(graph.numValues()),
      persistent_(graph.numValues()),
      moveable_output_mask_(graph.userOutputs().size()) {
<<<<<<< HEAD
  updatePersistentValues(/* weights = nullptr */);
  updateMovableOutputs();
}

ExecutionFrame::ExecutionFrame(
    const Graph& graph,
    const Weights& weights,
    const torch::nativert::ExecutorConfig& cfg,
    LayoutPlanner* layoutPlanner)
    : ExecutionFrame(graph) {
  setWeights(weights);
  if (layoutPlanner != nullptr) {
    layoutPlanner_ = layoutPlanner;
    layoutManager_ = std::make_unique<LayoutManager>(
        *layoutPlanner,
        *this,
        cfg.layoutPlannerSettings.layoutManagerSettings());
  }
}

void ExecutionFrame::setWeights(const Weights& weights) {
  weightVersion_ = weights.version();
  updatePersistentValues(&weights);
  updateMovableOutputs();
}

/* static */ std::vector<std::pair<ValueId, c10::IValue>> ExecutionFrame::
    getPersistentValues(const Graph& graph, const Weights* weights) {
  std::vector<std::pair<ValueId, c10::IValue>> persistentValues;

  /* ADD GRAPH-DEPENDENT PERSISTENT VALUES */

  for (const auto& [valueId, constSymintValue] :
       graph.getConstantSymIntValues()) {
    persistentValues.emplace_back(valueId, constSymintValue);
  }

  if (weights == nullptr) {
    return persistentValues;
  }

  /* ADD WEIGHT-DEPENDENT PERSISTENT VALUES */

  const auto& inputsToWeights = graph.signature().inputsToWeights();
  for (const auto& [inputName, weightName] : inputsToWeights) {
    const Value* value = graph.getValue(inputName);
    persistentValues.emplace_back(value->id(), weights->at(weightName));
  }

  const auto& inputsToCustomObjs = graph.signature().inputsToCustomObjs();
  for (const auto& [inputName, customObjName] : inputsToCustomObjs) {
    const Value* value = graph.getValue(inputName);
    persistentValues.emplace_back(
        value->id(), weights->getCustomObj(customObjName));
  }

  std::unordered_map<std::string, ValueId> foldedConstIds;
  for (const Node& node : graph.nodes()) {
=======
  // load constant SymInts into execution frame
  for (const auto& [valueId, constSymintValue] :
       graph_.getConstantSymIntValues()) {
    setPersistentIValue(valueId, constSymintValue);
  }

  for (const Node& node : graph_.nodes()) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    if (node.target() == "torch.ops.higher_order.run_const_graph") {
      const auto& const_graph =
          std::get<std::unique_ptr<Graph>>(node.attributes().at(0).value);
      for (size_t i = 0; i < node.outputs().size(); ++i) {
<<<<<<< HEAD
        foldedConstIds[std::string{const_graph->outputs().at(i)->name()}] =
=======
        foldedConstIds_[std::string{const_graph->outputs().at(i)->name()}] =
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            node.outputs()[i]->id();
      }
    }
  }
<<<<<<< HEAD
  for (const auto& [name, tensor] : weights->getFoldedConsts()) {
    persistentValues.emplace_back(foldedConstIds.at(name), tensor);
  }

  for (const auto& [name, iv] : weights->getConstFoldedValues()) {
    const Value* value = graph.getValue(name);
    persistentValues.emplace_back(value->id(), iv);
  }

  return persistentValues;
}

void ExecutionFrame::updatePersistentValues(const Weights* weights) {
  auto persistentValues = ExecutionFrame::getPersistentValues(graph_, weights);
  for (auto it = std::make_move_iterator(persistentValues.begin());
       it != std::make_move_iterator(persistentValues.end());
       ++it) {
    auto&& [value, iv] = *it;
    setPersistentIValue(value, std::move(iv));
  }
=======
}

ExecutionFrame::ExecutionFrame(const Graph& graph, const Weights& weights)
    : ExecutionFrame(graph) {
  setWeights(weights);
}

void ExecutionFrame::setWeights(const Weights& weights) {
  weightVersion_ = weights.version();

  const auto& inputsToWeights = graph_.signature().inputsToWeights();
  for (const auto& [inputName, weightName] : inputsToWeights) {
    const Value* value = graph_.getValue(inputName);
    setPersistentIValue(value->id(), weights.at(weightName));
  }

  const auto& inputsToCustomObjs = graph_.signature().inputsToCustomObjs();
  for (const auto& [inputName, customObjName] : inputsToCustomObjs) {
    const Value* value = graph_.getValue(inputName);
    setPersistentIValue(value->id(), weights.getCustomObj(customObjName));
  }

  for (const auto& [value, tensor] : weights.getFoldedConsts()) {
    setPersistentIValue(foldedConstIds_.at(value), tensor);
  }

  for (const auto& [n, iv] : weights.getConstFoldedValues()) {
    const Value* v = graph_.getValue(n);
    setPersistentIValue(v->id(), iv);
  }

  updateMovableOutputs();
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

void ExecutionFrame::updateMovableOutputs() {
  moveable_output_mask_.assign(moveable_output_mask_.size(), true);

  c10::FastSet<ValueId> inputs;
  for (const auto* input : graph_.userInputs()) {
    if (input) {
      inputs.insert(input->id());
    }
  }

  const auto& outputs = graph_.userOutputs();
  const size_t num_outputs = outputs.size();

  c10::FastSet<ValueId> seen;
  for (size_t i = 0; i < num_outputs; i++) {
    auto idx = num_outputs - 1 - i;
    if (const Value* const* valuePtr = std::get_if<Value*>(&outputs[idx]);
        valuePtr && *valuePtr) {
      auto id = (*valuePtr)->id();

      /*
          values are not moveable if:
          1. they are persistent
          2. they are inputs (since inputs are borrowed)
          3. the value will be moved in a later (right-more) output
      */

      if (!seen.insert(id).second || persistent_[id] ||
          inputs.find(id) != inputs.end()) {
        moveable_output_mask_[idx] = false;
      }
    }
  }
}

ExecutionFrame::ExecutionFrame(
    const Graph& graph,
    size_t numValues,
    const std::vector<ValueId>&,
    const std::vector<ValueId>&)
    : graph_(graph) {
  allValues_.resize(numValues);
}

void ExecutionFrame::setIValue(ValueId id, c10::IValue ivalue) {
  DCHECK(static_cast<size_t>(id) < allValues_.size());
  allValues_[id] = std::move(ivalue);
}

void ExecutionFrame::setBorrowedIValue(ValueId id, c10::IValue ivalue) {
  DCHECK(static_cast<size_t>(id) < allValues_.size());
  borrowedValueIds_.push_back(id);
  allValues_[id] = std::move(ivalue);
}

at::Tensor ExecutionFrame::getTensor(ValueId id) const {
  const auto& ivalue = getIValue(id);
  if (C10_LIKELY(ivalue.isTensor())) {
    return ivalue.toTensor();
  } else {
    throw std::runtime_error("getTensor called on non-tensor value");
  }
}

std::vector<c10::IValue> ExecutionFrame::tryMoveUserOutputs() {
  std::vector<c10::IValue> ret;
  const auto& outputs = graph_.userOutputs();
  ret.reserve(outputs.size());
  for (const auto& [i, outputValue] : c10::enumerate(outputs)) {
    if (const Value* const* valuePtr = std::get_if<Value*>(&outputValue);
        valuePtr && *valuePtr) {
      ret.push_back(
          isOutputMovable(i) ? moveIValue((*valuePtr)->id())
                             : getIValue((*valuePtr)->id()));
    } else if (Constant const* constant = std::get_if<Constant>(&outputValue)) {
      ret.push_back(constantToIValue(*constant));
    }
  }
  return ret;
}

} // namespace torch::nativert
