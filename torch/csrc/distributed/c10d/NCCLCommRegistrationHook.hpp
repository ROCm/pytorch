// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Bridges an NCCL process-group backend to a consumer that needs to track the
// backend's host communicator (today: symmetric memory's NCCLDevCommManager),
// without the backend depending on that consumer's headers. The consumer
// installs the hooks; the backend fires publishNCCLComm / retireNCCLComm on
// comm create / destroy. `comm` is an opaque host ncclComm_t, cast back on the
// consumer side, so this header stays free of nccl.h. Both entry points are
// no-ops until hooks are installed, so a build without symmetric-memory support
// -- or a comm created before the consumer is ever used -- simply does nothing.

#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

#include <chrono>
#include <functional>
#include <string>

namespace c10d {

// Lets consumers distinguish normal shutdown, where device resources can be
// drained and reclaimed, from abort/revoke paths that must retire them without
// waiting on potentially failed device work.
enum class NCCLCommRetirementMode {
  Graceful,
  Abort,
};

struct NCCLCommRegistrationHooks {
  std::function<void(void* comm)> on_initialize;
  std::function<void(const std::string& group_name, void* comm, c10::Device)>
      on_register;
  std::function<void(
      const std::string& group_name,
      void* comm,
      c10::Device,
      NCCLCommRetirementMode,
      std::chrono::milliseconds)>
      on_unregister;
};

// Installed once (typically at load time) by the consumer of the comm.
TORCH_API void setNCCLCommRegistrationHooks(NCCLCommRegistrationHooks hooks);

// Fired by comm producers; no-op if no hooks are installed. Producers must call
// retireNCCLComm while the host communicator is still valid.
TORCH_API void noteNCCLCommInitialized(void* comm);
TORCH_API void publishNCCLComm(
    const std::string& group_name,
    void* comm,
    c10::Device device);
TORCH_API void retireNCCLComm(
    const std::string& group_name,
    void* comm,
    c10::Device device,
    NCCLCommRetirementMode mode,
    std::chrono::milliseconds timeout);

} // namespace c10d
