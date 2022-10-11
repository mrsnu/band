#include "tensorflow/lite/planner/thermal_aware_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void ThermalAwareScheduler::Schedule(JobQueue& requests) {
  std::set<int> idle_workers = planner_->GetIdleWorkers();
  auto worker_id = rand() % idle_workers.size();
  if (!requests.empty()) {
    auto available_job = std::find_if(
        requests.begin(), requests.end(), [this, worker_id](const Job& job) {
          return GetInterpreter()->GetSubgraphIdx(job.model_id, worker_id) != -1;
        });
    if (available_job != requests.end()) {
      Job to_execute = *available_job;
      int subgraph_idx =
          GetInterpreter()->GetSubgraphIdx(to_execute.model_id, worker_id);
      Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
      EnqueueAction(to_execute, subgraph);

      requests.erase(available_job);
    }
  }
}

int64_t ThermalAwareScheduler::GetCurrentTemperature() {
  // TODO : implement
  return INT_MAX;
}

void ThermalAwareScheduler::UpdateExpectedLatency(JobQueue& requests,
                                                     int window_size) {
  // TODO : implement
}

void ThermalAwareScheduler::UpdateExpectedHeatGeneration(JobQueue& requests,
                                                     int window_size) {
  // TODO : implement
}

}  // namespace impl
}  // namespace tflite
