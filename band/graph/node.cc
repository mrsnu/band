#include "band/graph/node.h"

#include "band/graph/graph_builder.h"

namespace band {

std::shared_ptr<Node> BasicOp(TensorFunction func,
                              std::shared_ptr<Node> operand, std::string name) {
  return operand->builder()->AddBasicNode(func, operand, name);
}

std::shared_ptr<Node> ModelOp(const Model& model, std::shared_ptr<Node> operand,
                              std::string name) {
  return operand->builder()->AddModelNode(model, operand, name);
}

std::shared_ptr<Node> ModelOp(BackendType backend, std::string model_path,
                              std::shared_ptr<Node> operand, std::string name) {
  return operand->builder()->AddModelNodeFromPath(backend, model_path, operand,
                                                  name);
}

}  // namespace band