#include "band/graph/graph_builder.h"

#include "band/graph/graph.h"

namespace band {

bool GraphBuilder::IsValid() const { return true; }

absl::StatusOr<Graph> GraphBuilder::Build() {
  absl::StatusOr<Graph> ret;
  if (!IsValid()) {
    return absl::InternalError("Graph is not valid.");
  }

  std::set<size_t> no_output_nodes;
  for (auto& edge : edges_) {
    no_output_nodes.insert(edge.second);
  }
  for (auto& edge : edges_) {
    no_output_nodes.erase(edge.first);
  }

  for (auto& node : no_output_nodes) {
    edges_.push_back(Edge(node, nodes_[1]->id()));
  }
  return Graph(name_, nodes_, edges_);
}

Node* GraphBuilder::AddModelNode(Model model, Node* operand, std::string name) {
  nodes_.push_back(std::shared_ptr<Node>(new ModelNode(this, nodes_.size(), model, name)));
  edges_.push_back(Edge(operand->id(), nodes_.back()->id()));
  return nodes_.back().get();
};

Node* GraphBuilder::AddModelNodeFromPath(BackendType backend, std::string model_path,
                                         Node* operand, std::string name) {
  nodes_.push_back(std::shared_ptr<Node>(
      new ModelNode(this, nodes_.size(), backend, model_path, name)));
  edges_.push_back(Edge(operand->id(), nodes_.back()->id()));
  return nodes_.back().get();
};

Node* GraphBuilder::AddBasicNode(std::function<Tensors(Tensors)> func, Node* operand, std::string name) {
  nodes_.push_back(std::shared_ptr<Node>(new BasicNode(this, nodes_.size(), func, name)));
  edges_.push_back(Edge(operand->id(), nodes_.back()->id()));
  return nodes_.back().get();
}

}  // namespace band
