#ifndef BAND_MODEL_GRAPH_GRAPH_BUILDER_H_
#define BAND_MODEL_GRAPH_GRAPH_BUILDER_H_

#include <string>

#include "band/graph/node.h"
#include "band/model.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace band {

class Graph;
class Node;

using Edge = std::pair<size_t, size_t>;

class GraphBuilder {
 public:
  GraphBuilder(std::string name) : name_(name) {
    nodes_.push_back(std::make_shared<EntryNode>(this, 0, "Entry"));
    nodes_.push_back(std::make_shared<ExitNode>(this, 1, "Exit"));
  }
  bool IsValid() const;
  absl::StatusOr<Graph> Build();

  Node* AddModelNode(Model model, Node* operand, std::string name = "");
  Node* AddModelNodeFromPath(BackendType backend, std::string model_path,
                             Node* operand, std::string name = "");
  Node* AddBasicNode(std::function<Tensors(Tensors)> func, Node* operand,
                     std::string name = "");

  Node* GetEntryNode() { return nodes_[0].get(); }
  Node* GetExitNode() { return nodes_[1].get(); }

 private:
  std::string name_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<Edge> edges_;
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_GRAPH_BUILDER_H_