#ifndef BAND_TOOL_BENCHMARK_GRAPH_H_
#define BAND_TOOL_BENCHMARK_GRAPH_H_

#include <mutex>

#include "absl/status/statusor.h"
#include "band/config.h"

namespace band {
namespace tool {
class EngineRunnerConfig;
class BenchmarkGraph {
 public:
  static absl::StatusOr<BenchmarkGraph*> Create(
      const EngineRunnerConfig& config);

  struct Vertex {
    ModelId model_id;
    RequestOption request_option;
    size_t batch_size;
    size_t vertex_id;
  };

  void InitializeContexts();
  std::vector<Vertex> GetNextVertices() const;
  void OnVertexFinished(size_t vertex_id);
  bool IsFinished() const;

 private:
  BenchmarkGraph() = default;

  BenchmarkGraph(const BenchmarkGraph&);
  BenchmarkGraph& operator=(const BenchmarkGraph&) = default;

  std::set<size_t> GetResolvedVertexIds() const;
  absl::Status CheckCycles() const;

  mutable std::mutex mutex_;
  std::set<size_t> finished_vertices_;

  std::vector<Vertex> vertices_;
  std::vector<std::pair<size_t /* from */, size_t /* to */>> edges_;

};  // namespace tool

}  // namespace tool
}  // namespace band

#endif