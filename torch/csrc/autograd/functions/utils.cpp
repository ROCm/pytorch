#include <c10/util/irange.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <sstream>
#include <utility>

#include <execinfo.h>
#include <cxxabi.h>
#include <iostream>
#include <cstdlib>
#include <memory>

 
#include "execinfo.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <unistd.h>
#include <set>
#include <thread>
#include <mutex>

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <string>

#if 0
#define CCA_DEBUG(x) x
#else
#define CCA_DEBUG(x)
#endif

static std::string CcaGetEnv(const char* name, const char* default_value) {
  auto rtn = std::getenv(name);
  if (rtn) {
    return rtn;
  }
  return default_value;
}

static int dev_idx_to_print = -1;

static std::unordered_map<std::string, int> g_stack2id;
static int g_next_id = 1;
static std::mutex g_stack_mu;

static std::string frame_to_string(void* addr) {
  Dl_info info;
  if (!(dladdr(addr, &info) && info.dli_fname)) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%p ??", addr);
    return std::string(buf);
  }

  const char* so = info.dli_fname;
  const char* sym = info.dli_sname ? info.dli_sname : "??";
  void* sym_addr = info.dli_saddr;

#ifdef __GNUG__
  int status = 0;
  char* dem = abi::__cxa_demangle(sym, nullptr, nullptr, &status);
  const char* pretty = (status == 0 && dem) ? dem : sym;
#else
  const char* pretty = sym;
  char* dem = nullptr;
#endif

  uintptr_t base = (uintptr_t)info.dli_fbase;
  uintptr_t mod_off = base ? ((uintptr_t)addr - base) : 0;
  uintptr_t sym_off = sym_addr ? ((uintptr_t)addr - (uintptr_t)sym_addr) : 0;

  std::ostringstream oss;
  oss << addr << "  " << so
      << "  mod+0x" << std::hex << mod_off
      << "  " << pretty << " +0x" << std::hex << sym_off;

#ifdef __GNUG__
  std::free(dem);
#endif
  return oss.str();
}

static std::string make_stack_string(int skip = 0, int max_frames = 32) {
  void* frames[128];
  int n = backtrace(frames, std::min(max_frames, 128));

  std::ostringstream oss;
  for (int i = skip; i < n; ++i) {
    oss << frame_to_string(frames[i]) << "\n";
  }
  return oss.str();
}

int GetTraceID(bool force_print_trace, int skip, int max_frames) {
  CCADEBUG(if(0)) return 0;
  if (CcaGetEnv("DBGENV_PRINT_CALLSTACK", "0") == "0" && !force_print_trace) {
    return 0;
  }
  std::string stack = make_stack_string(skip, max_frames);

  int id = 0;
  bool first_time = false;
  {
    std::lock_guard<std::mutex> lk(g_stack_mu);
    auto it = g_stack2id.find(stack);
    if (it == g_stack2id.end()) {
      id = g_next_id++;
      g_stack2id.emplace(stack, id);
      first_time = true;
    } else {
      id = it->second;
    }
  }

  if (first_time) {
    std::fprintf(stderr, "[GetTraceID %d] new\n%s", id, stack.c_str());
  // } else {
  //   std::fprintf(stderr, "[STACK_ID %d]\n", id);
  }
  return id;
}

namespace torch::autograd::stream_tag {

thread_local std::vector<bool> tag_stack;

void push() { tag_stack.push_back(true); }
void pop() { if (!tag_stack.empty()) tag_stack.pop_back(); }
bool active() { return !tag_stack.empty() && tag_stack.back(); }

} // namespace


namespace torch::ddp_model2_stream {

Registry& registry() {
  static Registry r;
  return r;
}

} // namespace torch::ddp_model2_stream


namespace torch::autograd {

variable_list wrap_outputs(
    const variable_list& inputs,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    tensor_list&& outputs,
    const function_constructor& ctr) {
  variable_list result;
  result.reserve(outputs.size());
  if (!any_variable_requires_grad(inputs)) {
    for (auto& output : outputs) {
      if (output.defined()) {
        result.push_back(
            make_variable(std::move(output), /*requires_grad=*/false));
      } else {
        result.emplace_back();
      }
    }
  } else {
    auto grad_fn =
        ctr(GradMode::is_enabled() ? collect_next_edges(inputs) : edge_list());
    for (auto& output : outputs) {
      if (output.defined()) {
        auto variable =
            autograd::make_variable(std::move(output), /*requires_grad=*/false);
        autograd::create_gradient_edge(variable, grad_fn);
        result.push_back(std::move(variable));
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
        result.emplace_back();
      }
    }
  }
  return result;
}

void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args,
    bool allow_undefined) {
  if (required_args == -1) {
    required_args = args;
  }
  if (inputs.size() != static_cast<size_t>(args)) {
    std::stringstream ss;
    ss << name << ": expected " << args << " arguments (got " << inputs.size();
    ss << ")";
    throw std::runtime_error(ss.str());
  }
  for (const auto i : c10::irange(required_args)) {
    if (!inputs[i].defined() && !allow_undefined) {
      std::stringstream ss;
      ss << name << ": expected Tensor at argument " << i << " (got None)";
      throw std::runtime_error(ss.str());
    }
  }
}
} // namespace torch::autograd
