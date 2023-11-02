#include "band/graph/invariant.h"

#include <set>
#include <stack>

#include "band/graph/graph_builder.h"

namespace band {

bool NoCycleInvariant::Check(const GraphBuilder& graph) const {
  // Return false if cycle detected.
  std::stack<size_t> stack;
  std::set<size_t> visited;

  stack.push(0);
  while (!stack.empty()) {
    size_t node = stack.top();
    stack.pop();
    if (visited.find(node) != visited.end()) {
      return false;
    }
    visited.insert(node);
    for (auto& edge : graph.edges()) {
      if (edge.first == node) {
        stack.push(edge.second);
      }
    }
  }
  return true;
}

bool NoIsolatedNodeInvariant::Check(const GraphBuilder& graph) const {
  std::set<size_t> nodes;
  for (auto& edge : graph.edges()) {
    nodes.insert(edge.first);
    nodes.insert(edge.second);
  }
  for (auto& node : graph.nodes()) {
    nodes.erase(node->id());
  }
  if (nodes.size() > 0) {
    return false;
  }
  return true;
}

bool NoDuplicateEdgeInvariant::Check(const GraphBuilder& graph) const {
  std::set<Edge> unique_edges;
  for (auto& edge : graph.edges()) {
    if (unique_edges.find(edge) != unique_edges.end()) {
      return false;
    }
    unique_edges.insert(edge);
  }
  return true;
}

bool NoMismatchedEdgeInvariant::Check(const GraphBuilder& graph) const {
  for (auto& edge : graph.edges()) {
    if (edge.first >= graph.nodes().size() || edge.second >= graph.nodes().size()) {
      return false;
    }
  }
  return true;
}

}  // namespace band