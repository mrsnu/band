#ifndef BAND_TOOL_BENCHMARK_H_
#define BAND_TOOL_BENCHMARK_H_
#include <memory>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/tool/benchmark_config.h"

namespace band {
namespace tool {
class Benchmark {
 public:
  Benchmark(BackendType target_backend = BackendType::kTfLite);
  ~Benchmark();
  absl::Status Initialize(int argc, const char** argv);
  absl::Status Run();

 private:
  struct ModelContext {
    ~ModelContext();
    // simulate input tensor copy from model_inputs to model_request_inputs
    absl::Status PrepareInput();

    Model model;
    // pre-allocated model tensors for runtime requests
    std::vector<ModelId> model_ids;
    std::vector<RequestOption> request_options;
    std::vector<Tensors> model_request_inputs;
    std::vector<Tensors> model_request_outputs;
    // randomly generated input
    Tensors model_inputs;
  };

  const std::vector<double> cpu_frequencies = {
    0.8256,
    // 0.9408,
    // 1.0560,
    // 1.1712,
    // 1.2864,
    // 1.4016,
    // 1.4976,
    // 1.6128,
    // 1.7088,
    // 1.8048,
    1.9200,
    // 2.0160,
    // 2.1312,
    // 2.2272,
    // 2.3232,
    // 2.4192,
    // 2.5344,
    // 2.6496,
    // 2.7456,
    2.8416
  };

  const std::vector<double> gpu_frequencies = {
    0.5850,
    0.4992,
    0.4270, 
    0.3450, 
    0.2570
  };

  const std::vector<double> runtime_frequencies = {
    0.7104, 
    // 0.8256, 
    // 0.9408, 
    // 1.0560, 
    // 1.1712, 
    // 1.2864,
    // 1.4016, 
    // 1.4976, 
    1.6128, 
    // 1.7088, 
    // 1.8048, 
    // 1.9200,
    // 2.0160, 
    // 2.1312, 
    // 2.2272, 
    // 2.3232, 
    2.4192
  };

  // initialization
  bool ParseArgs(int argc, const char** argv);
  bool LoadBenchmarkConfigs(const Json::Value& root);
  bool LoadRuntimeConfigs(const Json::Value& root);

  // runner
  void RunPeriodic();
  void RunStream();
  void RunWorkload();
  void RunCPU();
  void RunGPU();
  void RunDSP();
  void RunNPU();
  void RunAll();

  absl::Status LogResults();

  const BackendType target_backend_;
  BenchmarkConfig benchmark_config_;
  RuntimeConfig* runtime_config_ = nullptr;
  std::unique_ptr<Engine> engine_ = nullptr;
  std::vector<ModelContext*> model_contexts_;
  bool kill_app_ = false;
};
}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_BENCHMARK_H_