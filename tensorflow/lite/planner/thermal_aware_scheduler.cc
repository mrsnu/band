#include "tensorflow/lite/planner/thermal_aware_scheduler.h"

#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

void ThermalAwareScheduler::Schedule(JobQueue& requests) {
  while (!requests.empty()) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = planner_->GetIdleAllWorkers();
    if (idle_workers.empty()) {
      continue;
    }

    Job job = requests.front();
    requests.pop_front();

    WorkerWaitingTime waiting_time = GetWorkerWaitingTime();
    std::pair<int, double> best_subgraph = GetMaxPptSubgraphIdx(job, waiting_time);

    Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);

    // job.estimated_temp = model_manager_->GetPredictedTemperature(
    //   target_subgraph->GetKey().worker_id, target_subgraph);
    // job.estimated_latency = model_manager_->GetPredictedLatency(
    //   target_subgraph->GetKey().worker_id, target_subgraph);
    EnqueueAction(job, target_subgraph);
  }
}

std::pair<int, double> ThermalAwareScheduler::GetMaxPptSubgraphIdx(Job& job, std::map<int, int64_t>& worker_waiting) {
  double max_ppt = -DBL_MAX;
  int max_idx = -1;

  std::vector<int> subgraph_indices = GetInterpreter()->GetSubgraphIndices(job.model_id);
  for (auto subgraph_index : subgraph_indices) {
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_index);
    SubgraphKey& key = subgraph->GetKey();

    int64_t waiting_time = worker_waiting[key.worker_id];
    std::pair<int, int64_t> source = model_manager_->GetPredictedTempAndLatency(key.worker_id, subgraph);
    int64_t expected_latency = source.second;
    int64_t total = expected_latency + waiting_time;
    thermal_t temp_diff = source.first;
    if (total <= 0) {
      total = 1; // epsilon value
    }
    if (temp_diff <= 0) {
      temp_diff = 1; // epsilon value
    }
    // LOGI("temp_diff= %d", temp_diff);

    double thermal_efficiency = 1 / (double)temp_diff * (double)1000;
    double ppt = (1.0 - eta_) * thermal_efficiency - eta_ * (double)total; // Note that this can be negatvie value!
    // double ppt = (eta_ - 1.0) * (double)temp_diff - eta_ * (double)total;
    // LOGI("thermal_Efficiency = %f", thermal_efficiency);
    // LOGI("ppt = %f", ppt);
    // job.estimated_ppt.push_back(ppt);
    // job.estimated_temp_diff.push_back(temp_diff);
    // job.estimated_total_latency.push_back(total);

    if (max_ppt < ppt) {
      max_ppt = ppt;
      max_idx = subgraph_index;
    }
  }
  return { max_idx, max_ppt };
}

}  // namespace impl
}  // namespace tflite
