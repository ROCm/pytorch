<<<<<<< HEAD
#include <torch/csrc/export/example_upgraders.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/export/pybind.h>
#include <torch/csrc/export/upgrader.h>
=======
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/export/pybind.h>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::_export {

void initExportBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto exportModule = rootModule.def_submodule("_export");
  auto pt2ArchiveModule = exportModule.def_submodule("pt2_archive_constants");

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExportedProgram>(exportModule, "CppExportedProgram");

  exportModule.def(
      "deserialize_exported_program", [](const std::string& serialized) {
<<<<<<< HEAD
        auto parsed = nlohmann::json::parse(serialized);

        // Query the current Python schema version as target
        // TODO: expose schema_version in gneerated_serialization_types.h and
        // access it here directly.
        py::module_ schema_module =
            py::module_::import("torch._export.serde.schema");
        py::tuple schema_version_tuple = schema_module.attr("SCHEMA_VERSION");
        int target_version = schema_version_tuple[0].cast<int>();

        auto upgraded = upgrade(parsed, target_version);
        return upgraded.get<ExportedProgram>();
=======
        return nlohmann::json::parse(serialized).get<ExportedProgram>();
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
      });

  exportModule.def("serialize_exported_program", [](const ExportedProgram& ep) {
    return nlohmann::json(ep).dump();
  });

<<<<<<< HEAD
  exportModule.def(
      "upgrade", [](const std::string& serialized_json, int target_version) {
        auto parsed = nlohmann::json::parse(serialized_json);
        auto upgraded = upgrade(parsed, target_version);
        return upgraded.dump();
      });

  exportModule.def(
      "register_example_upgraders", []() { registerExampleUpgraders(); });

  exportModule.def(
      "deregister_example_upgraders", []() { deregisterExampleUpgraders(); });

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  for (const auto& entry : torch::_export::archive_spec::kAllConstants) {
    pt2ArchiveModule.attr(entry.first) = entry.second;
  }
}
} // namespace torch::_export
