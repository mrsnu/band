#include "band/logger.h"
#include "band/tool/benchmark.h"

using namespace band;

int main(int argc, const char** argv) {
  tool::Benchmark benchmark;
  if (benchmark.Initialize(argc, argv).ok()) {
    benchmark.Run();
  } else {
    BAND_LOG_ERROR("Benchmark failed to initialize");
    return -1;
  }
  return 0;
}
