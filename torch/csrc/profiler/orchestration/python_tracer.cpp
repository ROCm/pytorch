#include <torch/csrc/profiler/orchestration/python_tracer.h>

namespace torch::profiler::impl::python_tracer {
namespace {
MakeFn make_fn;
<<<<<<< HEAD
=======
MakeMemoryFn memory_make_fn;
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

struct NoOpPythonTracer : public PythonTracerBase {
  NoOpPythonTracer() = default;
  ~NoOpPythonTracer() override = default;

  void stop() override {}
  void restart() override {}
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)>,
      std::vector<CompressedEvent>&,
      c10::time_t) override {
    return {};
  }
};
<<<<<<< HEAD
=======

struct NoOpMemoryPythonTracer : public PythonMemoryTracerBase {
  NoOpMemoryPythonTracer() = default;
  ~NoOpMemoryPythonTracer() override = default;
  void start() override {}
  void stop() override {}
  void export_memory_history(const std::string&) override {}
};

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace

void registerTracer(MakeFn make_tracer) {
  make_fn = make_tracer;
}

std::unique_ptr<PythonTracerBase> PythonTracerBase::make(RecordQueue* queue) {
  if (make_fn == nullptr) {
    return std::make_unique<NoOpPythonTracer>();
  }
  return make_fn(queue);
}
<<<<<<< HEAD
=======

void registerMemoryTracer(MakeMemoryFn make_memory_tracer) {
  memory_make_fn = make_memory_tracer;
}

std::unique_ptr<PythonMemoryTracerBase> PythonMemoryTracerBase::make() {
  if (memory_make_fn == nullptr) {
    return std::make_unique<NoOpMemoryPythonTracer>();
  }
  return memory_make_fn();
}
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
} // namespace torch::profiler::impl::python_tracer
