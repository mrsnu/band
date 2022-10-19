#include "tensorflow/lite/planner/thermal_aware_scheduler.h"

#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

void ThermalAwareScheduler::Schedule(JobQueue& requests) {
  std::set<int> idle_workers = planner_->GetIdleWorkers();
  // LOGI("Idle worker size : %d", idle_workers.size());
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();
    int model_id = to_execute.model_id;

    // Select a worker
    // std::vector<worker_id_t> possible_workers = model_manager_.GetPossibleWorkers(to_execute);
    int target_idx = rand() % idle_workers.size();
    std::set<int>::iterator it = idle_workers.begin();
    std::advance(it, target_idx);
    int worker_id = *it;
    // LOGI("It's selected : %d", worker_id);

    // Get a subgraph to execute
    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker_id);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);

    EnqueueAction(to_execute, subgraph);
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
