#include <string>

#include <torch/csrc/utils/pybind.h>

#ifdef USE_ROCM
#include <roctx.h>
#endif

namespace torch::cuda::shared {

#ifdef USE_ROCM

void initRoctxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto roctx = m.def_submodule("_roctx", "ROCTX bindings for ROCm profiling");

  roctx.def(
      "rangePushA",
      [](const std::string& msg) {
        return roctxRangePushA(msg.c_str());
      },
      py::arg("msg"));
  roctx.def("rangePop", []() { return roctxRangePop(); });
  roctx.def(
      "rangeStartA",
      [](const std::string& msg) {
        return static_cast<int64_t>(roctxRangeStartA(msg.c_str()));
      },
      py::arg("msg"));
  roctx.def(
      "rangeEnd",
      [](int64_t range_id) {
        roctxRangeStop(static_cast<roctx_range_id_t>(range_id));
      },
      py::arg("range_id"));
  roctx.def("markA", [](const std::string& msg) { roctxMarkA(msg.c_str()); }, py::arg("msg"));

  // ROCTX has no stream-callback API; stub to match NVTX API surface
  roctx.def(
      "deviceRangeStart",
      [](const std::string& /* msg */, std::intptr_t /* stream */) {
        return py::none();
      },
      py::arg("msg"),
      py::arg("stream") = 0);
  roctx.def(
      "deviceRangeEnd",
      [](py::object /* handle */, std::intptr_t /* stream */) {},
      py::arg("range_handle"),
      py::arg("stream") = 0);
}

#else

void initRoctxBindings(PyObject* module) {
  (void)module;
  // No-op when not ROCm: _roctx submodule is not registered,
  // so torch.cuda.roctx will get ImportError and use a stub.
}

#endif

} // namespace torch::cuda::shared
