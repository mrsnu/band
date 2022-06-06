#include "tensorflow/lite/planner/offloading_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void OffloadingScheduler::Schedule(JobQueue& requests) {
  JobQueue local_jobs;
  int window_size = std::min(planner_->GetWindowSize(), (int)requests.size());
  local_jobs.insert(local_jobs.begin(), requests.begin(),
                    requests.begin() + window_size);
  requests.erase(requests.begin(), requests.begin() + window_size);

  // schedule all jobs to offloading
  for (auto job : local_jobs) {
    int model_id = job.model_id;
    int worker_id = GetInterpreter()->GetRepresentativeWorkerId(kTfLiteOffloading);
    int subgraph_idx = GetInterpreter()->GetSubgraphIdx(model_id, worker_id);
    Subgraph* subgraph = GetInterpreter()->subgraph(subgraph_idx);
    EnqueueAction(job, subgraph);
  }
}

}  // namespace impl
}  // namespace tflite
