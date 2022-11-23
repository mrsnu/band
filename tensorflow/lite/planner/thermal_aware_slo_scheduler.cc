#include "tensorflow/lite/planner/thermal_aware_slo_scheduler.h"

#include <float.h>

#include "tensorflow/lite/profiling/time.h"
#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

void ThermalAwareSloScheduler::Schedule(JobQueue& requests) {
  while (!requests.empty()) {
    planner_->UpdateWorkerWaitingTime();
    std::set<int> idle_workers = planner_->GetIdleAllWorkers();
    if (idle_workers.empty()) {
      continue;
    }

    Job job = requests.front();
    requests.pop_front();
    
    WorkerWaitingTime waiting_time = GetWorkerWaitingTime();
    std::pair<int, double> best_subgraph = GetMinSloCostSubgraphIdx(job, waiting_time, job.slo_us);

    Subgraph* target_subgraph = GetInterpreter()->subgraph(best_subgraph.first);

    job.estimated_temp = model_manager_->GetPredictedTemperature(
      target_subgraph->GetKey().worker_id, target_subgraph);
    job.estimated_latency = model_manager_->GetPredictedLatency(
      target_subgraph->GetKey().worker_id, target_subgraph);
    EnqueueAction(job, target_subgraph);
  }
}

std::pair<int, double> ThermalAwareSloScheduler::GetMinSloCostSubgraphIdx(Job& job, std::map<int, int64_t> worker_waiting, int64_t slo_us) {
  double min_cost = DBL_MAX;
  int min_idx = -1;
  
  std::vector<int> subgraph_indices = GetInterpreter()->GetSubgraphIndices(job.model_id);
  for (auto subgraph_index : subgraph_indices) {
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_index);
    SubgraphKey& key = subgraph->GetKey();

    int64_t waiting_time = worker_waiting[key.worker_id];
    std::pair<int, int64_t> source = model_manager_->GetPredictedTempAndLatency(key.worker_id, subgraph);
    int64_t expected_latency = source.second;
    int64_t latency = expected_latency + waiting_time;
    thermal_t temp_diff = source.first;
    if (latency <= 0) {
      latency = 1;
    }

    if (temp_diff <= 0) {
      temp_diff = 1;
    }

    double eta = 0.5;
    // \Delta temp + \eta * max(0.f, SLO - FPS)
    LOGI("temp_diff: %d\n", temp_diff);
    LOGI("slo_us - latency: %lld\n", slo_us - latency);
    double slo_cost = temp_diff + eta * std::max(0L, slo_us - latency);
    LOGI("slo_cost = %lf\n", slo_cost);

    job.estimated_slo_cost.push_back(slo_cost);
    job.estimated_temp_diff.push_back(temp_diff);
    job.estimated_total_latency.push_back(latency);

    if (min_cost > slo_cost) {
      min_cost = slo_cost;
      min_idx = subgraph_index;
    }
  }
  return { min_idx, min_cost };
}

}  // namepsace impl
}  // namespace tflite