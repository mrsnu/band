#include "band/scheduler/dfs_graph_scheduler.h"

#include <stack>
#include <set>

namespace band {

std::vector<Job> DFSGraphScheduler::Schedule(Graph graph) {
  auto nodes = graph.nodes();
  auto edges = graph.edges();

  std::vector<Job> jobs;
  std::vector<size_t> order;
  std::set<size_t> visited;
  std::stack<size_t> stack;
  stack.push(0);
  while (!stack.empty()) {
    auto node_id = stack.top();
    stack.pop();
    if (visited.find(node_id) != visited.end()) {
      order.push_back(node_id);
      visited.insert(node_id);
    }
    for (auto edge : edges) {
      if (edge.first == node_id) {
        stack.push(edge.second);
      }
    }
  }
  return jobs;
}

}  // namespace band