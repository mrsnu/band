#include <thread>
#include <vector>

#include "band/engine.h"
#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_config.h"
#include "band/tool/benchmark_profiler.h"
#include "band/tool/runner.h"

namespace band {
namespace tool {
class GraphRunner : public IRunner {
 public:
  GraphRunner(BackendType target_backend, Engine& engine)
      : target_backend_(target_backend), engine_(engine) {}
  ~GraphRunner();
  virtual absl::Status Initialize(const Json::Value& root) override;
  virtual absl::Status Run() override;
  virtual void Join() override;
  virtual absl::Status LogResults(size_t instance_id) override;

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

  // runner
  void RunInternal();
  void RunPeriodic();
  void RunStream();
  void RunWorkload();

  std::thread runner_thread_;

  const BackendType target_backend_;
  Engine& const engine_;
  GraphRunnerConfig config_;
  bool kill_app_ = false;
  std::vector<ModelContext*> model_contexts_;
};
}  // namespace tool
}  // namespace band
