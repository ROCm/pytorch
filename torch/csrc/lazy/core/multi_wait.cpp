<<<<<<< HEAD
#include <c10/util/Exception.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <torch/csrc/lazy/core/multi_wait.h>

#include <chrono>
#include <exception>
#include <stdexcept>

namespace torch::lazy {

void MultiWait::Done() {
  bool notify = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_count_ += 1;
    notify = completed_count_ == count_;
  }
  if (notify) {
    cv_.notify_all();
  }
}

void MultiWait::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return completed_count_ >= count_; });
  if (exptr_ != nullptr) {
    std::rethrow_exception(exptr_);
  }
}

void MultiWait::Wait(double wait_seconds) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!cv_.wait_for(lock, std::chrono::duration<double>(wait_seconds), [this] {
        return completed_count_ >= count_;
      })) {
<<<<<<< HEAD
    TORCH_CHECK(false, "Timeout");
=======
    throw std::runtime_error("Timeout");
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  }
  if (exptr_ != nullptr) {
    std::rethrow_exception(exptr_);
  }
}

void MultiWait::Reset(size_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  count_ = count;
  completed_count_ = 0;
  exptr_ = nullptr;
}

std::function<void()> MultiWait::Completer(std::function<void()> func) {
  auto completer = [this, func = std::move(func)]() { Complete(func); };
  return completer;
}

std::function<void()> MultiWait::Completer(
    std::shared_ptr<MultiWait> mwait,
    std::function<void()> func) {
  auto completer = [mwait = std::move(mwait), func = std::move(func)]() {
    mwait->Complete(func);
  };
  return completer;
}

void MultiWait::Complete(const std::function<void()>& func) {
  try {
    func();
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex_);
    exptr_ = std::current_exception();
  }
  Done();
}

} // namespace torch::lazy
