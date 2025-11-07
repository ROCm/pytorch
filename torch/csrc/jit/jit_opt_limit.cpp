#include <cstdlib>
<<<<<<< HEAD
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
=======
#include <sstream>
#include <string>
#include <utility>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
<<<<<<< HEAD
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/jit_opt_limit.h>

=======
#include <c10/util/env.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/jit_opt_limit.h>

// NOTE: Don't try to migrate jit to C++17 yet
// As it's used in some embedded platforms

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
namespace torch::jit {

static std::unordered_map<std::string, int64_t>& passes_to_current_counter() {
  static std::unordered_map<std::string, int64_t> passes_to_current_counter;
  return passes_to_current_counter;
}

static int parseOptLimit(const std::string& opt_limit) {
  try {
    return std::stoi(opt_limit);
  } catch (...) {
    return -1;
  }
}

static std::unordered_map<std::string, int64_t> parseJITOptLimitOption(
<<<<<<< HEAD
    const char* option) {
  std::stringstream in_ss;
  if (option) {
    in_ss << option;
  }
=======
    const std::string& option) {
  std::stringstream in_ss;
  in_ss << option;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  std::unordered_map<std::string, int64_t> passes_to_opt_limits;
  std::string line;
  while (std::getline(in_ss, line, ':')) {
    if (line.empty()) {
      continue;
    }
    auto index_at = line.find_last_of('=');
    auto pass_name = line.substr(0, index_at);
    pass_name = c10::detail::ExcludeFileExtension(pass_name);
    auto opt_limit = parseOptLimit(line.substr(index_at + 1));
<<<<<<< HEAD
    passes_to_opt_limits.insert({pass_name, opt_limit});
=======
    passes_to_opt_limits.emplace(std::move(pass_name), opt_limit);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  }

  return passes_to_opt_limits;
}

bool opt_limit(const char* pass_name) {
<<<<<<< HEAD
  static const char* opt_limit = std::getenv("PYTORCH_JIT_OPT_LIMIT");
  // if nothing is provided, let's allow everything
  if (!opt_limit) {
=======
  static const auto opt_limit = c10::utils::get_env("PYTORCH_JIT_OPT_LIMIT");
  // if nothing is provided, let's allow everything
  if (!opt_limit.has_value()) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    return true;
  }

  static const std::unordered_map<std::string, int64_t> passes_to_opt_limits =
<<<<<<< HEAD
      parseJITOptLimitOption(opt_limit);
=======
      parseJITOptLimitOption(opt_limit.value());
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  std::string pass{pass_name};
  pass = c10::detail::StripBasename(pass);
  pass = c10::detail::ExcludeFileExtension(pass);

  auto opt_limit_it = passes_to_opt_limits.find(pass);
  if (opt_limit_it == passes_to_opt_limits.end()) {
    return true;
  }

  auto current_count_it = passes_to_current_counter().find(pass);
  if (current_count_it == passes_to_current_counter().end()) {
    passes_to_current_counter().insert({pass, 0});
  }

  current_count_it = passes_to_current_counter().find(pass);
  if (current_count_it->second >= opt_limit_it->second) {
    return false;
  }

  current_count_it->second++;
  return true;
}

} // namespace torch::jit
