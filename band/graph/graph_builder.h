#ifndef BAND_MODEL_GRAPH_GRAPH_BUILDER_H_
#define BAND_MODEL_GRAPH_GRAPH_BUILDER_H_

#include <string>

#include "band/graph/graph.h"
#include "band/graph/invariant.h"
#include "band/model.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace band {

using Edge = std::pair<size_t, size_t>;

class GraphBuilder : public IGraph {
 public:
  GraphBuilder(std::string name) : name_(name) {
    nodes_.push_back(std::make_shared<EntryNode>(this, 0, "Entry"));
    nodes_.push_back(std::make_shared<ExitNode>(this, 1, "Exit"));

    SetDefaultInvariants();
  }
  bool IsValid() const;
  
  absl::StatusOr<Graph> Build();

  std::shared_ptr<Node> AddModelNode(Model model, std::shared_ptr<Node> operand,
                                     std::string name = "");
  std::shared_ptr<Node> AddModelNodeFromPath(BackendType backend,
                                             std::string model_path,
                                             std::shared_ptr<Node> operand,
                                             std::string name = "");
  std::shared_ptr<Node> AddBasicNode(std::function<Tensors(Tensors)> func,
                                     std::shared_ptr<Node> operand,
                                     std::string name = "");
  void AddInvariant(std::unique_ptr<Invariant> invariant) {
    invariants_.push_back(std::move(invariant));
  }

  std::shared_ptr<Node> GetEntryNode() { return nodes_[0]; }
  std::shared_ptr<Node> GetExitNode() { return nodes_[1]; }

  std::vector<std::shared_ptr<Node>> nodes() const override { return nodes_; }
  std::vector<Edge> edges() const override { return edges_; }

 private:
  void SetDefaultInvariants() {
    invariants_.emplace_back(new NoCycleInvariant());
    invariants_.emplace_back(new NoIsolatedNodeInvariant());
    invariants_.emplace_back(new NoDuplicateEdgeInvariant());
    invariants_.emplace_back(new NoMismatchedEdgeInvariant());
  }

  std::string name_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<Edge> edges_;

  std::vector<std::unique_ptr<Invariant>> invariants_;
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_GRAPH_BUILDER_H_