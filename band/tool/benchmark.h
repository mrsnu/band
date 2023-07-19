#ifndef BAND_TOOL_BENCHMARK_H_
#define BAND_TOOL_BENCHMARK_H_
#include <memory>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_config.h"
#include "band/tool/benchmark_profiler.h"

namespace band {
namespace tool {

class BenchmarkInstance;
class Benchmark {
 public:
  Benchmark() = default;
  ~Benchmark();
  absl::Status Initialize(int argc, const char** argv);
  absl::Status Run();

 private:
  struct ModelContext {
    ~ModelContext();
    // simulate input tensor copy from model_inputs to model_request_inputs
    absl::Status PrepareInput();

    Model model;
    BenchmarkProfiler profiler;
    // pre-allocated model tensors for runtime requests
    std::vector<ModelId> model_ids;
    std::vector<RequestOption> request_options;
    std::vector<Tensors> model_request_inputs;
    std::vector<Tensors> model_request_outputs;
    // randomly generated input
    Tensors model_inputs;
  };

  // initialization
  bool ParseArgs(int argc, const char** argv);

  // runner
  void RunPeriodic();
  void RunStream();
  void RunWorkload();

  absl::Status LogResults();
  std::vector<BenchmarkInstance*> benchmark_instances_;
};

}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_BENCHMARK_H_