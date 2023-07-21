#ifndef BAND_TOOL_GRAPH_RUNNER_H_
#define BAND_TOOL_GRAPH_RUNNER_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "band/json_util.h"
#include "band/model.h"
#include "band/profiler.h"
#include "band/tool/benchmark_profiler.h"
#include "band/tool/runner.h"

namespace band {
namespace tool {

class BenchmarkGraph;
class EngineRunner;

class GraphRunner : public IRunner {
 public:
  GraphRunner(BackendType target_backend, EngineRunner& engine_runner)
      : target_backend_(target_backend), engine_runner_(engine_runner) {}
  virtual absl::Status Initialize(const Json::Value& root) override;
  virtual absl::Status Run() override;
  virtual void Join() override;
  virtual absl::Status LogResults(size_t instance_id) override;

 private:
  // runner
  void RunInternal();
  void RunPeriodic();
  void RunStream();
  void RunWorkload();

  const BackendType target_backend_;
  EngineRunner& const engine_runner_;
  BenchmarkProfiler profiler;

  std::string execution_mode_;
  size_t period_ms_;
  size_t slo_ms_;
  float slo_scale_;

  std::unique_ptr<BenchmarkGraph> graph_;
  std::thread runner_thread_;
  bool kill_app_ = false;
};
}  // namespace tool
}  // namespace band

#endif  // BAND_TOOL_GRAPH_RUNNER_H_