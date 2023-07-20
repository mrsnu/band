#ifndef BAND_MODEL_GRAPH_NODE_H_
#define BAND_MODEL_GRAPH_NODE_H_

#include <string>

#include "band/logger.h"
#include "band/model.h"
#include "band/tensor.h"

namespace band {

using Tensors = std::vector<interface::ITensor*>;
using TensorFunction = std::function<Tensors(Tensors)>;

using Dims = std::vector<size_t>;
using Shape = std::pair<DataType, Dims>;
using NodeInterface = std::pair<Shape, Shape>;

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

  size_t id() const { return id_; }
  GraphBuilder* builder() const { return builder_; }

  virtual NodeType GetType() = 0;
  DataType GetInputTensorType(size_t index) const;
  DataType GetOutputTensorType(size_t index) const;
  std::vector<int> GetInputTensorDims(size_t index) const;
  std::vector<int> GetOutputTensorDims(size_t index) const;

  bool IsConcrete() const {
    return input_tensor_type_ != DataType::kNoType &&
           output_tensor_type_ != DataType::kNoType &&
           !input_tensor_dims_.empty() && !output_tensor_dims_.empty();
  }

 private:
  GraphBuilder* builder_ = nullptr;
  size_t id_;
  std::string name_;
  DataType input_tensor_type_ = DataType::kNoType;
  DataType output_tensor_type_ = DataType::kNoType;
  std::vector<int> input_tensor_dims_ = {};
  std::vector<int> output_tensor_dims_ = {};
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
  BasicNode(GraphBuilder* builder, size_t id, TensorFunction func,
            std::string name = "")
      : Node(builder, id, name), func_(func) {}

  NodeType GetType() override { return NodeType::kBasic; }

  TensorFunction GetFunc() const { return func_; }

 private:
  TensorFunction func_;
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

std::shared_ptr<Node> BasicOp(TensorFunction func,
                              std::shared_ptr<Node> operand,
                              std::string name = "");
std::shared_ptr<Node> ModelOp(Model model, std::shared_ptr<Node> operand,
                              std::string name = "");
std::shared_ptr<Node> ModelOp(BackendType backend, std::string model_path,
                              std::shared_ptr<Node> operand,
                              std::string name = "");

}  // namespace band

#endif  // BAND_MODEL_GRAPH_NODE_H_