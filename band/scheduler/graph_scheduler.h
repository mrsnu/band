#ifndef BAND_SCHEDULER_GRAPH_SCHEDULER_H_
#define BAND_SCHEDULER_GRAPH_SCHEDULER_H_

#include "band/context.h"
#include "band/graph/graph.h"

namespace band {

class IGraphScheduler {
 public:
  explicit IGraphScheduler(Context& context) : context_(context) {}
  virtual ~IGraphScheduler() = default;

  virtual std::vector<Job> Schedule(Graph graph) = 0;
 protected:
  Context& context_;
};

}  // namespace band

#endif  // BAND_SCHEDULER_GRAPH_SCHEDULER_H_