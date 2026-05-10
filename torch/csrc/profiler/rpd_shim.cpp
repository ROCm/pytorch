#include <torch/csrc/profiler/rpd_shim.h>

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>

#include <c10/util/Exception.h>

namespace torch::profiler::impl::rpd {

namespace {

using RpdFn = void (*)();

struct RpdApi {
  RpdFn start = nullptr;
  RpdFn stop = nullptr;
  RpdFn flush = nullptr;
};

RpdApi g_api;
bool g_available = false;
std::once_flag g_init_flag;

void doInit() {
  // Check if already loaded (e.g. via LD_PRELOAD)
  void* handle = dlopen("librpd_tracer.so", RTLD_NOW | RTLD_NOLOAD);
  if (!handle) {
    // Disable auto-start so we control tracing via rpdstart/rpdstop.
    // Must be set before dlopen triggers Logger::rpdInit().
    setenv("RPDT_AUTOSTART", "0", 0);
    handle = dlopen("librpd_tracer.so", RTLD_NOW);
  }
  if (!handle) {
    return;
  }

  auto start = reinterpret_cast<RpdFn>(dlsym(handle, "rpdstart"));
  auto stop = reinterpret_cast<RpdFn>(dlsym(handle, "rpdstop"));
  auto flush = reinterpret_cast<RpdFn>(dlsym(handle, "rpdflush"));

  if (!start || !stop || !flush) {
    TORCH_WARN(
        "librpd_tracer.so loaded but missing symbols: ",
        !start ? "rpdstart " : "",
        !stop ? "rpdstop " : "",
        !flush ? "rpdflush " : "");
    return;
  }

  g_api = {start, stop, flush};
  g_available = true;
}

} // namespace

bool available() {
  std::call_once(g_init_flag, doInit);
  return g_available;
}

void prepareTrace() {
  std::call_once(g_init_flag, doInit);
}

void startTrace() {
  if (g_available) {
    g_api.start();
  }
}

void stopTrace() {
  if (g_available) {
    g_api.stop();
    g_api.flush();
  }
}

std::string traceFilePath() {
  const char* env = std::getenv("RPDT_FILENAME");
  return env ? std::string(env) : "./trace.rpd";
}

} // namespace torch::profiler::impl::rpd
