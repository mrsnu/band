#ifndef BAND_TOOL_ENGINE_RUNNER_H_
#define BAND_TOOL_ENGINE_RUNNER_H_
#include <memory>
#include <thread>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_config.h"
#include "band/tool/benchmark_profiler.h"
#include "band/tool/runner.h"

namespace band {
namespace tool {

struct ModelContext {
  ModelContext(std::unique_ptr<Model> model);
  ~ModelContext();
  // simulate input tensor copy from model_inputs to model_request_inputs
  absl::Status PrepareInput();

  std::unique_ptr<Model> model;
  BenchmarkProfiler profiler;
  // pre-allocated model tensors for runtime requests
  std::vector<ModelId> model_ids;
  std::vector<RequestOption> request_options;
  std::vector<Tensors> model_request_inputs;
  std::vector<Tensors> model_request_outputs;
  // randomly generated input
  Tensors model_inputs;
};

class EngineRunner : public IRunner {
 public:
  EngineRunner(BackendType target_backend = BackendType::kTfLite);
  virtual ~EngineRunner();

  virtual absl::Status Initialize(const Json::Value& root) override;
  virtual absl::Status Run() override;
  virtual void Join();
  virtual absl::Status LogResults(size_t instance_id);

  Engine& GetEngine() { return *engine_; }
  absl::StatusOr<Model&> GetModel(const std::string& model_name);

 private:
  absl::Status LoadRunnerConfigs(const Json::Value& root);
  absl::StatusOr<RuntimeConfig*> LoadRuntimeConfigs(const Json::Value& root);

  const BackendType target_backend_;

  size_t running_time_ms_;
  RuntimeConfig* runtime_config_ = nullptr;
  std::unique_ptr<Engine> engine_ = nullptr;

  std::mutex model_mutex_;
  std::map<std::string, Model*> registered_models_;
};

}  // namespace tool
}  // namespace band
#endif  // BAND_TOOL_BENCHMARK_INSTANCE_H_