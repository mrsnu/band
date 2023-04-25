#ifndef BAND_SCHEDULER_DFS_GRAPH_SCHEDULER_H_
#define BAND_SCHEDULER_DFS_GRAPH_SCHEDULER_H_

#include "band/scheduler/graph_scheduler.h"

namespace band {

class DFSGraphScheduler : public IGraphScheduler {
 public:
  explicit DFSGraphScheduler(Context& context) : IGraphScheduler(context) {}

  std::vector<Job> Schedule(Graph graph);
};

}  // namespace band

#endif  // BAND_SCHEDULER_DFS_GRAPH_SCHEDULER_H_