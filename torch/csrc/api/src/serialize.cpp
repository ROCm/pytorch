#include <torch/csrc/jit/serialization/pickle.h>
<<<<<<< HEAD
=======
#include <torch/csrc/jit/serialization/pickler.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <torch/serialize.h>

#include <vector>

namespace torch {

std::vector<char> pickle_save(const at::IValue& ivalue) {
  return jit::pickle_save(ivalue);
}

torch::IValue pickle_load(const std::vector<char>& data) {
  return jit::pickle_load(data);
}

} // namespace torch
