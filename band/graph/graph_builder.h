#ifndef BAND_MODEL_GRAPH_GRAPH_BUILDER_H_
#define BAND_MODEL_GRAPH_GRAPH_BUILDER_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/graph/graph.h"
#include "band/graph/invariant.h"
#include "band/model.h"

namespace band {

namespace {

std::vector<std::unique_ptr<Invariant>> GetDefaultInvariants() {
  std::vector<std::unique_ptr<Invariant>> invariants;
  invariants.push_back(std::make_unique<NoCycle>());
  invariants.push_back(std::make_unique<NoIsolatedNode>());
  invariants.push_back(std::make_unique<NoDuplicateEdge>());
  invariants.push_back(std::make_unique<NoMismatchedEdge>());
  return invariants;
}

}  // anonymous namespace

using Edge = std::pair<size_t, size_t>;

class GraphBuilder : public IGraph {
 public:
  GraphBuilder(std::string name) : name_(name) {
    nodes_.push_back(std::make_shared<EntryNode>(this, 0, "Entry"));
    nodes_.push_back(std::make_shared<ExitNode>(this, 1, "Exit"));
  }
  ~GraphBuilder();
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
    invariants_.push_back(invariant);
  }

  std::shared_ptr<Node> GetEntryNode() { return nodes_[0]; }
  std::shared_ptr<Node> GetExitNode() { return nodes_[1]; }

  std::vector<std::shared_ptr<Node>> nodes() const override { return nodes_; }
  std::vector<Edge> edges() const override { return edges_; }

 private:
  std::string name_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<Edge> edges_;

  std::vector<std::unique_ptr<Invariant>> invariants_ = GetDefaultInvariants();
};

}  // namespace band

#endif  // BAND_MODEL_GRAPH_GRAPH_BUILDER_H_