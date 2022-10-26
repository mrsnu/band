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
    std::vector<std::unique_ptr<Worker>>& workers = planner_->GetWorkers();

    Job to_execute = requests.front();
    int model_id = to_execute.model_id;

    int64_t earliest_finish_time = INT64_MAX;
    int64_t shortest_latency = INT64_MAX;
    Subgraph * target_subgraph;
    for (auto& worker : workers) {
      int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker->GetId());
      Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
      if (!model_manager_->IsAvailableWorker(worker->GetId(), subgraph)) {
        LOGI("[TAS] Throttling predicted! = Worker %d", worker->GetId());
        continue;
      }
      int64_t finish_time = worker->GetEstimatedFinishTime();
      int64_t latency = model_manager_->GetPredictedLatency(worker->GetId(), model_id);
      // LOGI("[TAS] worker %d finish_time = [%lld], latency = [%lld]", worker->GetId(), finish_time, latency);
      // LOGI("[TAS] earliest_finish_time = %lld value = %lld", earliest_finish_time, finish_time + latency);
      if (earliest_finish_time > finish_time + latency) {
        earliest_finish_time = finish_time + latency;
        shortest_latency = latency;
        target_subgraph = subgraph;
      }
    }

    if (earliest_finish_time == INT64_MAX) {
      LOGI("[TAS] All workers are throttled! => selects minimize throttled latency");
      for (auto& worker : workers) {
        int64_t finish_time = worker->GetEstimatedFinishTime();
        int64_t latency = model_manager_->GetPredictedThrottledLatency(worker->GetId(), model_id);
        if (earliest_finish_time > finish_time + latency) {
          earliest_finish_time = finish_time + latency;
          shortest_latency = latency;
          int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker->GetId());
          target_subgraph = GetInterpreter()->subgraph(subgraph_idx); 
        }
      } 
    }
    // insert estimated_latency, finish_time, temp 
    to_execute.estimated_latency = shortest_latency;
    to_execute.estimated_finish_time = earliest_finish_time;
    // to_execute.estimated_temp = model_manager_->GetPredictedTemperature(
    //   target_subgraph->GetKey().worker_id, target_subgraph, 
    //   workers[target_subgraph->GetKey().worker_id].get()->GetEstimatedEndTemperature());

    // LOGI("[Worker %d] selected!", target_subgraph->GetKey().worker_id);
    requests.pop_front();
    EnqueueAction(to_execute, target_subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
