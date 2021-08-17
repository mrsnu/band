#include "tensorflow/lite/planner/fixed_device_scheduler.h"
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void FixedDeviceGlobalQueueScheduler::Schedule(JobQueue& requests) {
  // TODO: fallback subgraphs for FixedDevicePlanner?
  std::set<int> idle_workers = planner_->GetIdleWorkers();
  if (idle_workers.empty()) {
    // no device is idle; wait for next iteration
    return;
  }

  for (auto it = requests.begin(); it != requests.end();) {
    Job& to_execute = *it;
    int model_id = to_execute.model_id;

    int worker_id;
    if (worker_id >= 0 && GetInterpreter()->GetNumWorkers()) {
      worker_id = to_execute.worker_id;
    } else {
      worker_id = planner_->GetModelWorkerMap()[model_id];
    }

    auto idle_workers_it = idle_workers.find(worker_id);
    if (idle_workers_it == idle_workers.end()) {
      // that device is not idle, so leave this job alone for now
      ++it;
      continue;
    }

    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker_id);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
    to_execute.expected_latency =
        GetWorkerWaitingTime()[worker_id] +
        GetInterpreter()->GetExpectedLatency(subgraph_idx);
    EnqueueAction(to_execute, subgraph);

    // delete this job from our request queue and
    // delete this device from our idle_workers set
    it = requests.erase(it);
    idle_workers.erase(idle_workers_it);

    if (idle_workers.empty()) {
      // no device is idle; wait for next iteration
      break;
    }
  }
}

}  // namespace impl
}  // namespace tflite
