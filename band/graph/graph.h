#ifndef BAND_MODEL_GRAPH_GRAPH_H_
#define BAND_MODEL_GRAPH_GRAPH_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/graph/node.h"
#include "band/model.h"

namespace band {

using Edge = std::pair<size_t, size_t>;

class IGraph {
 public:
  virtual std::vector<std::shared_ptr<Node>> nodes() const = 0;
  virtual std::vector<Edge> edges() const = 0;
  virtual std::vector<size_t> GetParents(size_t node_id) const;
  virtual std::vector<size_t> GetChildren(size_t node_id) const;
  virtual std::shared_ptr<Node> GetNodeById(size_t id) const {
    return nodes()[id];
  };
};

class Graph : public IGraph {
  friend class GraphBuilder;

 public:
  std::string GetGraphVizText() const;
  absl::Status SaveGraphViz(std::string path) const;

  std::vector<std::shared_ptr<Node>> nodes() const override { return nodes_; }
  std::vector<Edge> edges() const override { return edges_; }

  std::vector<size_t> GetTopologicalOrder() const;

 private:
  Graph(std::string name, std::vector<std::shared_ptr<Node>> nodes,
        std::vector<Edge> edges)
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