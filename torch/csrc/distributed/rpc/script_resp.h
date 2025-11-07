#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
<<<<<<< HEAD
=======
#include <torch/csrc/jit/serialization/pickler.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

namespace torch::distributed::rpc {

// Return value of a builtin operator or a TorchScript function.
class TORCH_API ScriptResp final : public RpcCommandBase {
 public:
  explicit ScriptResp(at::IValue&& values);

  const at::IValue& value();
  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptResp> fromMessage(const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::IValue value_;
};

} // namespace torch::distributed::rpc
