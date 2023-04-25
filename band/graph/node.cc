#include "band/graph/node.h"

#include "band/graph/graph_builder.h"

namespace band {

Node* BasicOp(std::function<Tensors(Tensors)> func, Node* operand, std::string name) {
  return operand->builder()->AddBasicNode(func, operand, name);
}

Node* ModelOp(Model model, Node* operand, std::string name) {
  return operand->builder()->AddModelNode(model, operand, name);
}

Node* ModelOp(BackendType backend, std::string model_path, Node* operand,
              std::string name) {
  return operand->builder()->AddModelNodeFromPath(backend, model_path, operand,
                                                  name);
}

}  // namespace band