#ifndef BAND_TOOL_BENCHMARK_H_
#define BAND_TOOL_BENCHMARK_H_
#include <memory>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_config.h"

namespace Band {
namespace tool {
class Benchmark {
 public:
  Benchmark(BackendType target_backend = BackendType::TfLite);
  ~Benchmark();
  BandStatus Initialize(int argc, const char** argv);
  BandStatus Run();

 private:
  struct ModelContext {
    ~ModelContext();
    // simulate input tensor copy from model_inputs to model_request_inputs
    BandStatus PrepareInput();

    Model model;
    Profiler profiler;
    // pre-allocated model tensors for runtime requests
    std::vector<ModelId> model_ids;
    std::vector<BandRequestOption> request_options;
    std::vector<Tensors> model_request_inputs;
    std::vector<Tensors> model_request_outputs;
    // randomly generated input
    Tensors model_inputs;
  };

  // initialization
  bool ParseArgs(int argc, const char** argv);
  bool LoadBenchmarkConfigs(const Json::Value& root);
  bool LoadRuntimeConfigs(const Json::Value& root);

  // runner
  void RunPeriodic();
  void RunStream();
  void RunWorkload();

  BandStatus LogResults();

  const BackendType target_backend_;
  BenchmarkConfig benchmark_config_;
  RuntimeConfig* runtime_config_ = nullptr;
  std::unique_ptr<Engine> engine_ = nullptr;
  std::vector<ModelContext*> model_contexts_;
  Profiler global_profiler_;
  bool kill_app_ = false;
};
}  // namespace tool
}  // namespace Band

#endif  // BAND_TOOL_BENCHMARK_H_