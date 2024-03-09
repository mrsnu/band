#ifndef BAND_GRAPH_GRAPH_H_
#define BAND_GRAPH_GRAPH_H_

#include "band/graph/graph_interface.h"
#include "band/graph/node.h"
#include "band/model.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace band {

class Graph : public IGraph {
  friend class GraphBuilder;

 public:
  std::string GetGraphVizText() const;
  absl::Status SaveGraphViz(std::string path) const;
  std::vector<size_t> GetTopologicalOrder() const;

 private:
  Graph(std::string name, std::vector<std::shared_ptr<Node>> nodes,
        std::vector<Edge> edges)
      : IGraph(name) {
    for (auto& node : nodes) {
      nodes_.push_back(node);
    }
    for (auto& edge : edges) {
      edges_.push_back(edge);
    }
  }
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_GRAPH_H_