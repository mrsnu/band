#include "tensorflow/lite/planner/mobile_cloud_lst_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void MobileCloudLstScheduler::Schedule(JobQueue& requests) {
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  if (window_size <= 0) {
    return;
  }

  std::set<int> idle_workers = planner_->GetIdleWorkers();
  if (idle_workers.empty()) {
    return;
  }

  planner_->UpdateWorkerWaitingTime();
  WorkerWaitingTime waiting_time = GetWorkerWaitingTime();

  int64_t current_time = profiling::time::NowMicros();
  SortBySlackTime(requests, window_size, current_time);

  std::set<int> job_indices_to_erase;
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    Job job = *it;

    std::pair<std::vector<int>, int> best_subgraph =
        GetInterpreter()->GetSubgraphWithShortestLatency(job, waiting_time);

    int target_subgraph_idx = best_subgraph.first.front();
    Subgraph* target_subgraph = GetInterpreter()->subgraph(target_subgraph_idx);
    if (job.slo_us > 0 &&
        current_time + best_subgraph.second > job.enqueue_time + job.slo_us) {
      job.status = kTfLiteJobSLOViolation;
      EnqueueAction(job, target_subgraph);
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }

    int worker_id = target_subgraph->GetKey().worker_id;
    if (idle_workers.find(worker_id) != idle_workers.end()) {
      waiting_time[worker_id] +=
          GetInterpreter()->GetExpectedLatency(target_subgraph_idx);
      EnqueueAction(job, target_subgraph);
      job_indices_to_erase.insert(it - requests.begin());
      continue;
    }
  }
  for (auto it = job_indices_to_erase.rbegin();
       it != job_indices_to_erase.rend(); ++it) {
    requests.erase(requests.begin() + *it);
  }
}

int64_t MobileCloudLstScheduler::GetSlackTime(int64_t current_time,
                                               const Job& job) {
  if (job.slo_us > 0) {
    int64_t deadline = job.enqueue_time + job.slo_us;
    int64_t remaining_execution_time = job.expected_latency;
    return deadline - current_time - remaining_execution_time;
  } else {
    return INT_MAX;
  }
}

void MobileCloudLstScheduler::SortBySlackTime(JobQueue& requests,
                                               int window_size,
                                               int64_t current_time) {
  UpdateExpectedLatency(requests, window_size);
  std::sort(requests.begin(), requests.begin() + window_size,
            [&](const Job& first, const Job& second) -> bool {
              return GetSlackTime(current_time, first) <
                     GetSlackTime(current_time, second);
            });
}

void MobileCloudLstScheduler::UpdateExpectedLatency(JobQueue& requests,
                                                     int window_size) {
  for (auto it = requests.begin(); it != requests.begin() + window_size; ++it) {
    it->expected_latency =
        GetInterpreter()
            ->GetSubgraphWithShortestLatency(*it, GetWorkerWaitingTime())
            .second;
  }
}

}  // namespace impl
}  // namespace tflite
