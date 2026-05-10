#pragma once

#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch {

TORCH_API void global_rlog_init();
TORCH_API void rlog_set_record_shapes(bool enable);
TORCH_API void rlog_set_record_stacks(bool enable);

using RlogStackCallback = void (*)(std::string&);
TORCH_API void rlog_set_stack_callback(RlogStackCallback cb);

} // namespace torch
