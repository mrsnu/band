#include "tensorflow/lite/planner/thermal_aware_scheduler.h"

#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

void ThermalAwareScheduler::Schedule(JobQueue& requests) {
  while (!requests.empty()) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = planner_->GetIdleWorkers();

    Job to_execute = requests.front();
    int model_id = to_execute.model_id;

    // Get available workers which is not throttled now and 
    // will not occur thermal throttling on executing the inference request
    int64_t shortest_latency = INT_MAX;
    Subgraph * target_subgraph;
    for (auto wid : idle_workers) {
      LOGI("[Worker %d] Idle", wid);
      int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, wid);
      Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
      if (!model_manager_->IsAvailableWorker(wid, subgraph)) {
        LOGI("[Worker %d] Throttling predicted!", wid);
        continue;
      }
      LOGI("[Worker %d] Available", wid);
      int64_t latency = model_manager_->GetPredictedLatency(wid, model_id);
      if (shortest_latency > latency) {
        shortest_latency = latency;
        target_subgraph = subgraph;
      }
    }
    if (shortest_latency == INT_MAX) {
      // When all workers are not available, 
      // the scheduler should wait until any worker becomes available.
      continue;
    }
    LOGI("[Worker %d] selected!", target_subgraph->GetKey().worker_id);
    requests.pop_front();
    EnqueueAction(to_execute, target_subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
