#ifndef BAND_MODEL_GRAPH_GRAPH_H_
#define BAND_MODEL_GRAPH_GRAPH_H_

#include "band/graph/node.h"
#include "band/model.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace band {

using Edge = std::pair<size_t, size_t>;

class Graph {
  friend class GraphBuilder;

 public:
  std::string GetGraphVizText();
  void SaveGraphViz(std::string path);

  std::vector<std::shared_ptr<Node>> nodes() { return nodes_; }
  std::vector<Edge> edges() { return edges_; }

 private:
  Graph(std::string name, std::vector<std::shared_ptr<Node>> nodes, std::vector<Edge> edges)
      : name_(name), edges_(edges) {
    for (auto& node : nodes) {
      nodes_.push_back(node);
    }
  }

  std::string name_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<Edge> edges_;
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_GRAPH_H_