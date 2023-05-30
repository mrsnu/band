#include "band/scheduler/shortest_expected_latency_scheduler.h"

#include <unordered_set>

#include "band/logger.h"
#include "band/time.h"

namespace band {
ShortestExpectedLatencyScheduler::ShortestExpectedLatencyScheduler(
    Context& context, int window_size)
    : IScheduler(context), window_size_(window_size) {}

bool ShortestExpectedLatencyScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  JobQueue local_jobs;
  int window_size = std::min(window_size_, (int)requests.size());
  local_jobs.insert(local_jobs.begin(), requests.begin(),
                    requests.begin() + window_size);
  requests.erase(requests.begin(), requests.begin() + window_size);
  while (!local_jobs.empty()) {
    context_.UpdateWorkersWaiting();
    // First, find the most urgent job -- the one with the
    // largest shortest latency (no, that's not a typo).
    // Put that job into some worker, and repeat this whole loop until we've
    // gone through all jobs.
    // There should be a more quicker way do this, but I'm leaving this as-is
    // to make it simple.
    // E.g., we add interpreter.GetProfiledLatency() to the expected_latency map
    // of all Jobs instead of calling GetShortestLatency() a gazillion times
    // again.

    // Note that we are NOT considering enqueue_time at the moment;
    // no request is given higher priority even if it had stayed in the queue
    // for longer than others.

    // find the most urgent job and save its index within the queue
    int64_t largest_shortest_latency = -1;
    int target_job_idx;
    SubgraphKey target_subgraph_key;
    WorkerWaitingTime worker_waiting = context_.GetWorkerWaitingTime();

    std::unordered_set<std::pair<int, BitMask>, JobIdBitMaskHash> searched_jobs;
    for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
      Job& next_job = *it;

      std::pair<int, BitMask> job_to_search =
          std::make_pair(next_job.model_id, next_job.resolved_unit_subgraphs);
      if (searched_jobs.find(job_to_search) != searched_jobs.end()) {
        continue;
      } else {
        searched_jobs.insert(job_to_search);
      }

      std::pair<std::vector<SubgraphKey>, int64_t> best_subgraph =
          context_.GetSubgraphWithShortestLatency(next_job, worker_waiting);

      if (largest_shortest_latency < best_subgraph.second) {
        largest_shortest_latency = best_subgraph.second;
        target_job_idx = it - local_jobs.begin();
        target_subgraph_key = best_subgraph.first.front();
      }
    }

    if (target_subgraph_key.IsValid() == false) {
      continue;
    }

    // for some reason, this Job must NOT be a reference (&), otherwise
    // we get a segfault at push_back() below
    Job most_urgent_job = local_jobs[target_job_idx];

    // remove the job from the queue so that we don't meet it in the next loop
    local_jobs.erase(local_jobs.begin() + target_job_idx);

    if (context_.IsBegin(most_urgent_job.subgraph_key)) {
      // only set these fields if this is the first subgraph of this model
      most_urgent_job.expected_latency = largest_shortest_latency;
    }
    success &= context_.EnqueueToWorker({most_urgent_job, target_subgraph_key});
  }
  return success;
}
}  // namespace band
