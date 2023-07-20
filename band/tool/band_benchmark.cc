#include "band/logger.h"
#include "band/tool/benchmark.h"

using namespace band;

int main(int argc, const char** argv) {
  band::tool::Benchmark benchmark;
  if (benchmark.Initialize(argc, argv).ok()) {
    auto status = benchmark.Run();
    if (!status.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Benchmark failed: %s", status.message());
      return -1;
    }
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Benchmark failed to initialize");
    return -1;
  }
  return 0;
}
