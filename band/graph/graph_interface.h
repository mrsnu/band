#ifndef BAND_GRAPH_GRAPH_INTERFACE_H_
#define BAND_GRPHA_GRAPH_INTERFACE_H_

#include "absl/status/status.h"

namespace band {

class Node;

using Edge = std::pair<size_t, size_t>;

class IGraph {
 public:
  IGraph(std::string name) : name_(name) {}
  std::vector<std::shared_ptr<Node>> nodes() const { return nodes_; };
  std::vector<Edge> edges() const { return edges_; };

  std::vector<size_t> GetParents(size_t node_id) const;
  std::vector<size_t> GetChildren(size_t node_id) const;
  std::shared_ptr<Node> GetNodeById(size_t id) const {
    return nodes()[id];
  };

 protected:
  std::string name_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<Edge> edges_;
};

}  // namespace band

#endif  // BAND_GRAPH_GRAPH_INTERFACE_H_