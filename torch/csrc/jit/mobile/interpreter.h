#pragma once

#include <vector>

#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/frame.h>

namespace torch::jit::mobile {

struct InterpreterState {
  TORCH_API explicit InterpreterState(const Code& code);
  TORCH_API bool run(Stack& stack);

 private:
<<<<<<< HEAD
  void enterFrame(const Code& /*code*/);
=======
  void enterFrame(const Code&);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  void leaveFrame();
  void saveExceptionDebugHandles();
  void callFunction(torch::jit::Function& f, Stack& stack);

  c10::IValue& reg(size_t reg);
  std::vector<c10::IValue> registers_;
  std::vector<Frame> frames_;
};

const std::vector<DebugHandle>& getInterpretersExceptionDebugHandles();
} // namespace torch::jit::mobile
