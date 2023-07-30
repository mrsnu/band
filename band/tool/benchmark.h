#ifndef BAND_TOOL_BENCHMARK_H_
#define BAND_TOOL_BENCHMARK_H_
#include <memory>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_profiler.h"
#include "band/tool/runner.h"

namespace band {
namespace tool {

class EngineRunner;
class Benchmark : public IRunner {
 public:
  Benchmark() = default;
  absl::Status Initialize(int argc, const char** argv);
  virtual absl::Status Run() override;

 private:
  // initialization
  bool ParseArgs(int argc, const char** argv);

  absl::Status LogResults();
};

}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_BENCHMARK_H_