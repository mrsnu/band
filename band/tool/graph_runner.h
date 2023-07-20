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
  GraphRunner(BackendType target_backend, EngineRunner& engine_runner)
      : target_backend_(target_backend), engine_runner_(engine_runner) {}
  ~GraphRunner();
  virtual absl::Status Initialize(const Json::Value& root) override;
  virtual absl::Status Run() override;
  virtual void Join() override;
  virtual absl::Status LogResults(size_t instance_id) override;

 private:
  struct Vertex {
    std::string path;
    size_t batch_size = 1;
    int worker_id = -1;
    size_t model_id;

    const RequestOption GetRequestOption() const {
      RequestOption option = RequestOption::GetDefaultOption();
      if (worker_id >= 0) {
        option.target_worker = worker_id;
      }
      return option;
    }
  };

  // runner
  void RunInternal();
  void RunPeriodic();
  void RunStream();
  void RunWorkload();

  std::thread runner_thread_;

  const BackendType target_backend_;
  EngineRunner& const engine_runner_;
  bool kill_app_ = false;

  std::vector<Vertex*> model_contexts_;
};
}  // namespace tool
}  // namespace band
