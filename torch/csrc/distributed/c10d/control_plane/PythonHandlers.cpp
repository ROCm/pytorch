#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

#include <cstdio>
#include <fstream>
#include <string>

<<<<<<< HEAD
#include <c10/util/Exception.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <c10/util/tempfile.h>
#include <torch/csrc/distributed/c10d/exception.h>
#include <torch/csrc/utils/pybind.h>

namespace c10d::control_plane {
namespace {

RegisterHandler tracebackHandler{
    "dump_traceback",
    [](const Request&, Response& res) {
      auto tmpfile = c10::make_tempfile("torch-dump_traceback");

      auto cfile = ::fopen(tmpfile.name.c_str(), "w");
<<<<<<< HEAD
      TORCH_CHECK(cfile, "failed to open file for writing");
=======
      if (!cfile) {
        throw std::runtime_error("failed to open file for writing");
      }
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

      {
        py::gil_scoped_acquire guard{};

        auto faulthandler = py::module::import("faulthandler");
        faulthandler.attr("dump_traceback")(fileno(cfile), true);
      }

      ::fclose(cfile);

      std::ifstream file(tmpfile.name);
      std::string str;
      std::string file_contents;
      while (std::getline(file, str)) {
        file_contents += str;
        file_contents.push_back('\n');
      }

      res.setContent(std::move(file_contents), "text/plain");
    }};
} // namespace
} // namespace c10d::control_plane
