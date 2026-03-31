#include "rlog_client.h"
#include "rlog/client.h"

#include <aten/src/ATen/record_function.h>

#include <torch/csrc/profiler/util.h>
#include <nlohmann/json.hpp>


using at::RecordFunction;
using at::RecordFunctionCallback;
using at::CallbackHandle;
using at::ObserverContext;

namespace {

  RecordFunctionCallback *cb = NULL;
  CallbackHandle handle = 0;

  // Rlog interface

  bool isLogging {false};
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
  Client client;
  //----------------

const char* categories[(uint8_t)at::RecordScope::NUM_SCOPES] = { "" };


std::unique_ptr<ObserverContext> enter_callback(const RecordFunction &func)
{
  if (isLogging) {
    nlohmann::json object;
    //object["name"] = func.name();
    //object["scope"] = categories[(uint8_t)func.scope()];
    object["seq"] = func.seqNr();
    object["op_id"] = func.handle();
    object["sizes"] = torch::profiler::impl::inputSizes(func, true);
    rlog::rangePush(categories[(uint8_t)func.scope()], func.name(), object.dump().c_str());
  }
  return NULL;
}

void exit_callback(const RecordFunction &func, ObserverContext *context)
{
  if (isLogging) {
    rlog::rangePop();
  }
}

} // namespace


namespace torch {

void global_rlog_init() {
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

  cb = new RecordFunctionCallback(enter_callback, exit_callback);
  cb->needsInputs(true);
  //cb->needsOutputs(true);
  cb->needsIds(true);

  // Register Callback and disable
  handle = at::addThreadLocalCallback(*cb);
  if (isLogging == false)
      at::disableCallback(handle);
}

} // namespace torch
