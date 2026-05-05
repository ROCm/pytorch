#include "rlog_client.h"
#include "rlog/client.h"

#include <aten/src/ATen/record_function.h>

#include <torch/csrc/profiler/util.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>


using at::RecordFunction;
using at::RecordFunctionCallback;
using at::CallbackHandle;
using at::ObserverContext;

namespace {

  std::unique_ptr<RecordFunctionCallback> cb;
  std::atomic<CallbackHandle> handle {0};

  // Rlog interface

  std::atomic<bool> isLogging {false};
  void rlog_callback_function() {
    isLogging = rlog::isActive();
    if (handle != 0) {
      if (isLogging)
        at::reenableCallback(handle);	
      else
        at::disableCallback(handle);
    }
  }

  class Client {
  public:
    Client() {
      rlog::init();
      rlog::registerActiveCallback(&rlog_callback_function);
      rlog::setDefaultDomain("torch");
      rlog::setDefaultCategory("");
    }
  };
  //----------------

const char* categories[(uint8_t)at::RecordScope::NUM_SCOPES] = { "" };

std::atomic<bool> record_shapes {false};
std::atomic<bool> record_input_op_ids {false};
std::once_flag init_flag;
std::mutex producer_map_mutex;
// Keyed by TensorImpl* cast to void*. No invalidation on tensor destruction; reused addresses
// overwrite stale entries. This matches the NVTX implementation and is an accepted limitation.
std::unordered_map<void*, std::pair<at::RecordFunctionHandle, int>> producer_tensor_map;


std::unique_ptr<ObserverContext> enter_callback(const RecordFunction &func)
{
  if (isLogging) {
    nlohmann::json object;
    //object["name"] = func.name();
    //object["scope"] = categories[(uint8_t)func.scope()];
    object["seq"] = func.seqNr();
    object["op_id"] = func.handle();
    if (record_shapes)
      object["sizes"] = torch::profiler::impl::inputSizes(func, true);
    if (record_input_op_ids) {
      nlohmann::json op_ids = nlohmann::json::array();
      std::lock_guard<std::mutex> lock(producer_map_mutex);
      for (const c10::IValue& input : func.inputs()) {
        if (input.isTensor()) {
          const at::Tensor& tensor = input.toTensor();
          if (tensor.defined()) {
            auto it = producer_tensor_map.find((void*)tensor.unsafeGetTensorImpl());
            if (it != producer_tensor_map.end())
              op_ids.push_back({it->second.first, it->second.second});
            else
              op_ids.push_back(nullptr);
          } else {
            op_ids.push_back(nullptr);
          }
        } else {
          op_ids.push_back(nullptr);
        }
      }
      object["input_op_ids"] = op_ids;
    }
    rlog::rangePush(categories[(uint8_t)func.scope()], func.name(), object.dump().c_str());
  }
  return NULL;
}

void exit_callback(const RecordFunction &func, ObserverContext *context)
{
  if (isLogging) {
    if (record_input_op_ids) {
      std::lock_guard<std::mutex> lock(producer_map_mutex);
      int output_nr = 0;
      for (const c10::IValue& output : func.outputs()) {
        if (output.isTensor()) {
          const at::Tensor& tensor = output.toTensor();
          if (tensor.defined())
            producer_tensor_map[(void*)tensor.unsafeGetTensorImpl()] = {func.handle(), output_nr};
        }
        output_nr++;
      }
    }
    rlog::rangePop();
  }
}

} // namespace


namespace torch {

void global_rlog_init() {
  std::call_once(init_flag, []() {
    static Client client;
    categories[(uint8_t)at::RecordScope::FUNCTION] = "function";
    categories[(uint8_t)at::RecordScope::BACKWARD_FUNCTION] = "backward_function";
    categories[(uint8_t)at::RecordScope::TORCHSCRIPT_FUNCTION] = "torchscript_function";
    categories[(uint8_t)at::RecordScope::KERNEL_FUNCTION_DTYPE] = "kernel_function_dtype";
    categories[(uint8_t)at::RecordScope::CUSTOM_CLASS] = "custom_class";
    categories[(uint8_t)at::RecordScope::BUILD_FEATURE] = "build_feature";
    categories[(uint8_t)at::RecordScope::LITE_INTERPRETER] = "lite_interpreter";
    categories[(uint8_t)at::RecordScope::USER_SCOPE] = "user_scope";
    categories[(uint8_t)at::RecordScope::STATIC_RUNTIME_OP] = "static_runtime_op";
    categories[(uint8_t)at::RecordScope::STATIC_RUNTIME_MODEL] = "static_runtime_model";

    auto propBool = [](const char* property, const char* defaultValue) {
      std::string val = rlog::getProperty("torch", property, defaultValue);
      std::transform(val.begin(), val.end(), val.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (val == "true")  return true;
      if (val == "false") return false;
      try { return std::stoi(val) != 0; } catch (...) { return false; }
    };

    record_shapes = propBool("record_shapes", "true");
    record_input_op_ids = propBool("record_input_op_ids", "false");
    cb = std::make_unique<RecordFunctionCallback>(enter_callback, exit_callback);
    cb->needsInputs(record_shapes || record_input_op_ids);
    cb->needsOutputs(record_input_op_ids);
    cb->needsIds(true);

    // Register Callback and disable
    handle = at::addGlobalCallback(*cb);
    if (isLogging == false)
        at::disableCallback(handle);
  });
}

} // namespace torch
