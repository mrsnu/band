#include "band/logger.h"
#include "band/tool/benchmark.h"

using namespace band;

int main(int argc, const char** argv) {
  band::tool::Benchmark benchmark;
  auto status = benchmark.Initialize(argc, argv);
  if (status.ok()) {
    auto status = benchmark.Run();
    if (!status.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Benchmark failed: %s", status.message());
      return -1;
    }
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Benchmark failed to initialize: %s", status.message());
    return -1;
  }
  return 0;
}
