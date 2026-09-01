// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/NCCLCommRegistrationHook.hpp>

#include <mutex>
#include <utility>

namespace c10d {

namespace {
std::mutex& hooksMutex() {
  static std::mutex m;
  return m;
}

NCCLCommRegistrationHooks& hooks() {
  static NCCLCommRegistrationHooks h;
  return h;
}
} // namespace

void setNCCLCommRegistrationHooks(NCCLCommRegistrationHooks h) {
  std::lock_guard<std::mutex> lock(hooksMutex());
  hooks() = std::move(h);
}

void noteNCCLCommInitialized(void* comm) {
  std::function<void(void*)> cb;
  {
    std::lock_guard<std::mutex> lock(hooksMutex());
    cb = hooks().on_initialize;
  }
  if (cb) {
    cb(comm);
  }
}

void publishNCCLComm(
    const std::string& group_name,
    void* comm,
    c10::Device device) {
  // Copy the callback out under the lock, then invoke it without the lock held
  // so the consumer's own locking (e.g. NCCLDevCommManager) can't deadlock.
  std::function<void(const std::string&, void*, c10::Device)> cb;
  {
    std::lock_guard<std::mutex> lock(hooksMutex());
    cb = hooks().on_register;
  }
  if (cb) {
    cb(group_name, comm, device);
  }
}

void retireNCCLComm(
    const std::string& group_name,
    void* comm,
    c10::Device device,
    NCCLCommRetirementMode mode,
    std::chrono::milliseconds timeout) {
  std::function<
      void(
          const std::string&,
          void*,
          c10::Device,
          NCCLCommRetirementMode,
          std::chrono::milliseconds)>
      cb;
  {
    std::lock_guard<std::mutex> lock(hooksMutex());
    cb = hooks().on_unregister;
  }
  if (cb) {
    cb(group_name, comm, device, mode, timeout);
  }
}

} // namespace c10d
