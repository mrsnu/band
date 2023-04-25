#include "band/graph/graph.h"

#include <fstream>

#include "absl/strings/str_format.h"

namespace band {

namespace {

std::string GetNodeIdString(Node* node) {
  switch (node->GetType()) {
    case NodeType::kEntry:
      return absl::StrFormat("entry_%d", node->id());
    case NodeType::kExit:
      return absl::StrFormat("exit_%d", node->id());
    case NodeType::kModel:
      return absl::StrFormat("model_%d", node->id());
    case NodeType::kBasic:
      return absl::StrFormat("basic_%d", node->id());
    default:
      return absl::StrFormat("unknown_%d", node->id());
  }
  return "";
}

std::string GetNodeAttribute(Node* node) {
  switch (node->GetType()) {
    case NodeType::kEntry:
      return absl::StrFormat(
          "label=%s, shape=box, style=filled, fillcolor=gray", node->GetName());
    case NodeType::kExit:
      return absl::StrFormat(
          "label=%s, shape=box, style=filled, fillcolor=gray", node->GetName());
    case NodeType::kModel:
      return absl::StrFormat(
          "label=%s, shape=box, style=filled, fillcolor=lightblue",
          node->GetName());
    case NodeType::kBasic:
      return absl::StrFormat(
          "label=%s, shape=box, style=filled, fillcolor=lightyellow",
          node->GetName());
    default:
      return absl::StrFormat(
          "label=%s, shape=box, style=filled, fillcolor=white",
          node->GetName());
  }
  return "";
}

}  // anonymous namespace

std::string Graph::GetGraphVizText() {
  std::string ret;
  ret += "digraph " + name_ + " {\n";
  ret += "  {\n";
  for (auto& node : nodes_) {
    ret += "  " + GetNodeIdString(node.get()) + " [" +
           GetNodeAttribute(node.get()) + "];\n";
  }
  ret += "  }\n";
  for (auto& edge : edges_) {
    ret += "  " + GetNodeIdString(nodes_[edge.first].get()) + " -> " +
           GetNodeIdString(nodes_[edge.second].get()) + ";\n";
  }
  ret += "}\n";
  return ret;
}

void Graph::SaveGraphViz(std::string path) {
  std::ofstream file(path);
  file << GetGraphVizText();
}

}  // namespace band