#include "band/graph/invariant.h"

#include <set>

#include "band/graph/graph_builder.h"

namespace band {
  
bool NoCycle::Check(const GraphBuilder graph) const {
  return true;
}

bool NoIsolatedNode::Check(const GraphBuilder graph) const {
  return true;
}

bool NoDuplicateEdge::Check(const GraphBuilder graph) const {
  std::set<Edge> unique_edges;
  for (auto& edge : graph.edges()) {
    if (unique_edges.find(edge) != unique_edges.end()) {
      return false;
    }
    unique_edges.insert(edge);
  }
  return true;
}

bool NoMismatchedEdge::Check(const GraphBuilder graph) const {
  return true;
}

}  // namespace band