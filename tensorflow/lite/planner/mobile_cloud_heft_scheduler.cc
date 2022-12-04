#include "tensorflow/lite/planner/mobile_cloud_heft_scheduler.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)
namespace tflite {
namespace impl {

void MobileCloudHeftScheduler::Schedule(JobQueue& requests) {
  while (!requests.empty()) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = planner_->GetIdleAllWorkers();
    if (idle_workers.empty()) {
      continue;
    }

    Job job = requests.front();
    requests.pop_front();

    WorkerWaitingTime waiting_time = GetWorkerWaitingTime();

    std::pair<int, int64_t> best_subgraph = GetShortestSubgraph(job.model_id, waiting_time);

    Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);

    // job.estimated_temp = model_manager_->GetPredictedTemperature(
    //   target_subgraph->GetKey().worker_id, target_subgraph);
    // job.estimated_latency = model_manager_->GetPredictedLatency(
    //   target_subgraph->GetKey().worker_id, target_subgraph);
    EnqueueAction(job, target_subgraph);
  }
}

std::pair<int, int64_t>
MobileCloudHeftScheduler::GetShortestSubgraph(int model_id, std::map<int, int64_t>& worker_waiting) {
  int64_t min_latency = INT64_MAX;
  int min_idx = -1;

  std::vector<int> subgraph_indices = GetInterpreter()->GetSubgraphIndices(model_id);
  for (auto subgraph_index : subgraph_indices) {
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_index);
    SubgraphKey& key = subgraph->GetKey();

    int64_t waiting_time = worker_waiting[key.worker_id];
    int64_t expected_latency = model_manager_->GetPredictedLatency(key.worker_id, subgraph);
    int64_t total = expected_latency + waiting_time;

    if (min_latency > total) {
      min_latency = total;
      min_idx = subgraph_index;
    }
  }
  return {min_idx, min_latency};
}


}  // namespace impl
}  // namespace tflite