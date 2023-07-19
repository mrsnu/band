#ifndef BAND_TOOL_BENCHMARK_INSTANCE_H_
#define BAND_TOOL_BENCHMARK_INSTANCE_H_
#include <memory>
#include <thread>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_config.h"
#include "band/tool/benchmark_profiler.h"

namespace band {
namespace tool {
class BenchmarkInstance {
 public:
  BenchmarkInstance(BackendType target_backend = BackendType::kTfLite);
  ~BenchmarkInstance();

  absl::Status Initialize(const Json::Value& root);

  absl::Status Run();
  void Join();
  absl::Status LogResults(size_t instance_id);

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
  absl::Status LoadBenchmarkConfigs(const Json::Value& root);
  absl::Status LoadRuntimeConfigs(const Json::Value& root);

  // runner
  void RunInternal();
  void RunPeriodic();
  void RunStream();
  void RunWorkload();

  std::thread runner_thread_;

  const BackendType target_backend_;
  BenchmarkInstanceConfig benchmark_config_;
  RuntimeConfig* runtime_config_ = nullptr;
  std::unique_ptr<Engine> engine_ = nullptr;
  std::vector<ModelContext*> model_contexts_;
  BenchmarkProfiler global_profiler_;
  bool kill_app_ = false;
};

}  // namespace tool
}  // namespace band
#endif  // BAND_TOOL_BENCHMARK_INSTANCE_H_