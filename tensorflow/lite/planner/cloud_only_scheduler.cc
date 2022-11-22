#include "tensorflow/lite/planner/cloud_only_scheduler.h"

namespace tflite {
namespace impl {

void CloudOnlyScheduler::Schedule(JobQueue& requests) {
  while (!requests.empty()) {
    int worker_id = kTfLiteCLOUD;

    Job to_execute = requests.front();
    requests.pop_front();
    int model_id = to_execute.model_id;

    // Get a subgraph to execute
    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker_id);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);

    EnqueueAction(to_execute, subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
