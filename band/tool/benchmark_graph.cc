#include "band/tool/benchmark_graph.h"

#include <algorithm>
#include <queue>

#include "band/tool/benchmark_config.h"

namespace band {
namespace tool {
BenchmarkGraph::BenchmarkGraph(const BenchmarkGraph& rhs)
    : vertices_(rhs.vertices_), edges_(rhs.edges_) {}

absl::StatusOr<BenchmarkGraph*> BenchmarkGraph::Create(
    const EngineRunnerConfig& config) {
  BenchmarkGraph* graph = new BenchmarkGraph();

  // Create vertices
  for (auto& model_config : config.model_configs) {
    BenchmarkGraph::Vertex vertex;
    vertex.model_id = model_config.model_id;
    vertex.request_option = model_config.GetRequestOption();
    vertex.batch_size = model_config.batch_size;
    vertex.vertex_id = graph->vertices_.size();
    graph->vertices_.push_back(vertex);
  }

  // Create edges
  for (auto& edge : config.edges) {
    graph->edges_.push_back(edge);
  }

  auto status = graph->CheckCycles();
  if (status != absl::OkStatus()) {
    delete graph;
    return status;
  }

  return graph;
}

void BenchmarkGraph::InitializeContexts() {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_vertices_.clear();
}

std::vector<BenchmarkGraph::Vertex> BenchmarkGraph::GetNextVertices() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Vertex> vertices;
  std::set<size_t> resolved_vertex_ids = GetResolvedVertexIds();
  std::set<size_t> executable_vertex_ids;
  // executable_vertex_ids = resolved_vertex_ids - finished_vertices_
  std::set_difference(
      resolved_vertex_ids.begin(), resolved_vertex_ids.end(),
      finished_vertices_.begin(), finished_vertices_.end(),
      std::inserter(executable_vertex_ids, executable_vertex_ids.begin()));
  for (auto vertex_id : executable_vertex_ids) {
    vertices.push_back(vertices_[vertex_id]);
  }
  return vertices;
}

void BenchmarkGraph::OnVertexFinished(size_t vertex_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_vertices_.insert(vertex_id);
}

bool BenchmarkGraph::IsFinished() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return finished_vertices_.size() == vertices_.size();
}

std::set<size_t> BenchmarkGraph::GetResolvedVertexIds() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::set<size_t> resolved_vertices;
  // A vertex is resolved if all its incoming edges are resolved
  for (size_t i = 0; i < vertices_.size(); ++i) {
    bool is_resolved = true;
    for (size_t j = 0; j < edges_.size(); ++j) {
      if (edges_[j].second == i && finished_vertices_.find(edges_[j].first) ==
                                       finished_vertices_.end()) {
        is_resolved = false;
        break;
      }
    }
    if (is_resolved) {
      resolved_vertices.insert(i);
    }
  }
  return resolved_vertices;
}

absl::Status BenchmarkGraph::CheckCycles() const {
  std::vector<size_t> num_incoming_edges(vertices_.size(), 0);
  std::queue<size_t> queue;
  size_t num_visited_vertices = 0;
  // Initialize num_incoming_edges per vertex
  for (size_t i = 0; i < edges_.size(); ++i) {
    ++num_incoming_edges[edges_[i].second];
  }
  // Start with vertices with no incoming edges
  for (size_t i = 0; i < num_incoming_edges.size(); ++i) {
    if (num_incoming_edges[i] == 0) {
      queue.push(i);
    }
  }
  // BFS
  while (!queue.empty()) {
    size_t vertex_id = queue.front();
    queue.pop();
    ++num_visited_vertices;

    for (size_t i = 0; i < edges_.size(); ++i) {
      if (edges_[i].first == vertex_id) {
        --num_incoming_edges[edges_[i].second];
        if (num_incoming_edges[edges_[i].second] == 0) {
          queue.push(edges_[i].second);
        }
      }
    }
  }

  return num_visited_vertices == vertices_.size()
             ? absl::OkStatus()
             : absl::InternalError("Cycles detected");
}
}  // namespace tool
}  // namespace band