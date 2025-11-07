#pragma once

#include <stdexcept>

#include <torch/csrc/Export.h>
#include <optional>
#include <string>

namespace torch::jit {

struct TORCH_API JITException : public std::runtime_error {
  explicit JITException(
      const std::string& msg,
      std::optional<std::string> python_class_name = std::nullopt,
      std::optional<std::string> original_msg = std::nullopt);

  std::optional<std::string> getPythonClassName() const {
    return python_class_name_;
  }

<<<<<<< HEAD
  // the original msg if this is from a python exception. The interpreter has
=======
  // the original msg if this is from a python exception. The interpretor has
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  // changed the original message by adding "The following operation failed in
  // the TorchScript interpreter." in front of it in the handleError function.
  std::optional<std::string> getOriginalMsg() const {
    return original_msg_;
  }

  static const std::string& getCaughtOriginalMsg();
  static const std::string& getCaughtPythonClassName();
  static void setCaughtOriginalMsg(const std::string& msg);
  static void setCaughtPythonClassName(const std::string& pythonClassName);

 private:
  std::optional<std::string> python_class_name_;
  std::optional<std::string> original_msg_;
};

} // namespace torch::jit
