#ifndef BAND_MODEL_GRAPH_NODE_H_
#define BAND_MODEL_GRAPH_NODE_H_

#include <string>

#include "band/logger.h"
#include "band/model.h"
#include "band/tensor.h"

namespace band {

using Tensors = std::vector<interface::ITensor*>;

class GraphBuilder;

enum class NodeType {
  kEntry = 0,
  kExit = 1,
  kBasic = 2,
  kModel = 3,
};

class Node {
 public:
  Node() = delete;
  Node(GraphBuilder* builder, size_t id, std::string name)
      : builder_(builder), id_(id), name_(name) {}
  std::string GetName() const { return name_; }
  virtual NodeType GetType() = 0;

  size_t id() const { return id_; }
  GraphBuilder* builder() const { return builder_; }

 private:
  size_t id_;
  std::string name_;
  GraphBuilder* builder_;
};

class EntryNode : public Node {
 public:
  EntryNode() = delete;
  EntryNode(GraphBuilder* builder, size_t id, std::string name)
      : Node(builder, id, name) {}
  NodeType GetType() override { return NodeType::kEntry; }
};

class ExitNode : public Node {
 public:
  ExitNode() = delete;
  ExitNode(GraphBuilder* builder, size_t id, std::string name)
      : Node(builder, id, name) {}
  NodeType GetType() override { return NodeType::kExit; }
};

class BasicNode : public Node {
 public:
  BasicNode() = delete;
  BasicNode(GraphBuilder* builder, size_t id,
            std::function<Tensors(Tensors)> func, std::string name = "")
      : Node(builder, id, name), func_(func) {}

  NodeType GetType() override { return NodeType::kBasic; }

  std::function<Tensors(Tensors)> GetFunc() const { return func_; }

 private:
  std::function<Tensors(Tensors)> func_;
};

class ModelNode : public Node {
 public:
  ModelNode() = delete;
  ModelNode(GraphBuilder* builder, size_t id, Model model,
            std::string name = "")
      : Node(builder, id, name), model_(model) {}
  ModelNode(GraphBuilder* builder, size_t id, BackendType backend,
            std::string model_path, std::string name = "")
      : Node(builder, id, name) {
    auto status = model_.FromPath(backend, model_path.c_str());
    if (!status.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to load model from path: %s",
                    model_path.c_str());
    }
  };

  NodeType GetType() override { return NodeType::kModel; }

  Model GetModel() const { return model_; }

 private:
  Model model_;
};

Node* BasicOp(std::function<Tensors(Tensors)> func, Node* operand,
              std::string name = "");
Node* ModelOp(Model model, Node* operand, std::string name = "");
Node* ModelOp(BackendType backend, std::string model_path, Node* operand,
              std::string name = "");

}  // namespace band

#endif  // BAND_MODEL_GRAPH_NODE_H_