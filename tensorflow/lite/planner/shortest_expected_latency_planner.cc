#include "tensorflow/lite/planner/shortest_expected_latency_planner.h"
#include "tensorflow/lite/profiling/time.h"

// for the std::cout commented out in Plan()
// #include <iostream>

namespace tflite {
namespace impl {

void ShortestExpectedLatencyPlanner::Plan() {
  int sched_id = 0;
  while (true) {
    if (GetSafeBool().wait()) return;

    std::deque<Job> local_jobs;
    std::unique_lock<std::mutex> request_lock(GetRequestsMtx());
    std::deque<Job>& requests = GetRequests();
    if (!requests.empty()) {
      // Gets the specific amount of jobs from requests
      // and removes those jobs from the requests.
      int window_size = std::min(GetWindowSize(), (int)requests.size());
      local_jobs.insert(local_jobs.begin(), requests.begin(),
                        requests.begin() + window_size);
      requests.erase(requests.begin(), requests.begin() + window_size);
    } else {
      continue;
    }
    request_lock.unlock();

    while (!local_jobs.empty()) {
      // First, find the most urgent job -- the one with the
      // largest shortest latency (no, that's not a typo).
      // Put that job into some worker, and repeat this whole loop until we've
      // gone through all jobs.
      // There should be a more quicker way do this, but I'm leaving this as-is
      // to make it simple.
      // E.g., we add interpreter.GetProfiledLatency() to the expected_latency
      // map of all Jobs instead of calling GetShortestLatency() a gazillion
      // times again.

      // Note that we are NOT considering enqueue_time at the moment;
      // no request is given higher priority even if it had stayed in the queue
      // for longer than others.

      // first, get per-device waiting times
      std::map<TfLiteDeviceFlags, int64_t> device_waiting_time;
      for (int i = 0; i < kTfLiteNumDevices; ++i) {
        TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
        Worker* worker = GetInterpreter()->GetWorker(device_flag);
        if (worker != nullptr) {
          device_waiting_time[device_flag] = worker->GetWaitingTime();
        }
      }

      // find the most urgent job and save its index within the queue
      int64_t largest_shortest_latency = -1;
      int target_job_idx;
      int target_subgraph_idx;

      int64_t sched_start = profiling::time::NowMicros();
      for (auto it = local_jobs.begin(); it != local_jobs.end(); ++it) {
        Job& next_job = *it;

        Subgraph* start_subgraph = interpreter_->subgraph(
            interpreter_->GetFirstSubgraphIdx(next_job.model_id, kTfLiteCPU));

        std::set<int> resolved_output;
        resolved_output.insert(start_subgraph->inputs().begin(), start_subgraph->inputs().end());

        std::pair<int, int64_t> best_subgraph =
            GetShortestLatency(next_job.model_id, resolved_output, 0, device_waiting_time);

        if (largest_shortest_latency < best_subgraph.second) {
          largest_shortest_latency = best_subgraph.second;
          target_job_idx = it - local_jobs.begin();
          target_subgraph_idx = best_subgraph.first;
        }
      }
      int64_t sched_end = profiling::time::NowMicros();
      // quick check for roughly examining the planning overhead
      // std::cout << "Time to Find the next job(us) : " <<  sched_end -
      // sched_start << std::endl;

      // for some reason, this Job must NOT be a reference (&), otherwise
      // we get a segfault at push_back() below
      Job most_urgent_job = local_jobs[target_job_idx];

      // remove the job from the queue so that we don't meet it in the next loop
      local_jobs.erase(local_jobs.begin() + target_job_idx);

      Subgraph* target_subgraph =
          GetInterpreter()->subgraph(target_subgraph_idx);
      SubgraphKey& to_execute = target_subgraph->GetKey();
      most_urgent_job.subgraph_idx = target_subgraph_idx;
      most_urgent_job.device_id = to_execute.device_flag;
      most_urgent_job.sched_id = sched_id++;

      ModelSpec& model_spec =
          GetInterpreter()->GetModelSpec(most_urgent_job.model_id);
      if (!target_subgraph->GetNextSubgraph()) {
        Job remaining_ops(most_urgent_job.model_id);
        remaining_ops.enqueue_time = most_urgent_job.enqueue_time;
        remaining_ops.following_jobs = most_urgent_job.following_jobs;

        most_urgent_job.following_jobs.clear();
        most_urgent_job.following_jobs.push_back(remaining_ops);
      }

      Worker* worker = GetInterpreter()->GetWorker(to_execute.device_flag);
      {
        std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
        worker->GetDeviceRequests().push_back(most_urgent_job);
        worker->GetRequestCv().notify_one();
      }
    }
  }
}

bool ShortestExpectedLatencyPlanner::NeedProfile() { return true; }

std::pair<int, int64_t> ShortestExpectedLatencyPlanner::GetShortestLatency(
    int model_id, std::set<int> resolved_output, int64_t start_time,
    std::map<TfLiteDeviceFlags, int64_t>& device_waiting,
    TfLiteDeviceFlags preceded_device) {
  std::vector<int> subgraph_indices =
      GetSubgraphCandidates(model_id, resolved_output, preceded_device);
  std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>
      subgraph_map = GroupByStartEndIdx(subgraph_indices);

  std::pair<int, int64_t> min_subgraph = {-1, INT_MAX};
  for (auto iter = subgraph_map.begin(); iter != subgraph_map.end(); iter++) {
    // first, filter out the subgraphs that take longer than others with the
    // same start/end indices, since there's no reason to pick them
    std::pair<int, int64_t> target_subgraph =
        GetShortestSubgraphIndex(iter->second, start_time, device_waiting);
    Subgraph* subgraph = interpreter_->subgraph(target_subgraph.first);
    SubgraphKey& key = subgraph->GetKey();

    std::set<int> next_resolved_output = resolved_output;
    
    for (const int& input_tensor : subgraph->inputs()) {
      next_resolved_output.erase(input_tensor);
    }

    std::copy(subgraph->outputs().begin(), subgraph->outputs().end(),
              std::inserter(next_resolved_output, next_resolved_output.begin()));

    std::pair<int, int64_t> local_min;
    if (next_resolved_output != interpreter_->GetModelSpec(model_id).output_tensors) {
      // there's more ops left for this model, so we need to look further to
      // get the final latency
      local_min =
          GetShortestLatency(model_id, next_resolved_output, target_subgraph.second,
                             device_waiting, key.device_flag);
    } else {
      // nothing else after this
      local_min = target_subgraph;
    }

    // check if this subgraph is better than the best one
    if (local_min.second < min_subgraph.second) {
      // note the subgraph to return is the next immediate one (start_idx, XX),
      // but the latency to return is that of the final subgraph (XX, #ops)
      // hence, target_subgraph.first & local_min.second
      min_subgraph.first = target_subgraph.first;
      min_subgraph.second = local_min.second;
    }
  }

  return min_subgraph;
}

std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>
ShortestExpectedLatencyPlanner::GroupByStartEndIdx(
    std::vector<int> subgraph_indices) {
  std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>> ret;
  for (auto subgraph_index : subgraph_indices) {
    SubgraphKey& key = interpreter_->subgraph(subgraph_index)->GetKey();
    ret[{key.input_ops, key.output_ops}].push_back(
        subgraph_index);
  }
  return ret;
}

std::vector<int> ShortestExpectedLatencyPlanner::GetSubgraphCandidates(
    int model_id, std::set<int> resolved_output, TfLiteDeviceFlags preceded_device) {
  std::vector<int> candidate_indices;

  // iterate thru all subgraphs and only pick the ones that match the criteria
  for (int i = 0; i < interpreter_->subgraphs_size(); ++i) {
    Subgraph* subgraph = interpreter_->subgraph(i);
    SubgraphKey& key = subgraph->GetKey();
    if (key.model_id == model_id &&
        key.device_flag != preceded_device) {
      bool is_executable = true;

      for (const int& input_tensor : subgraph->inputs()) {
        if (resolved_output.find(input_tensor) == resolved_output.end()) {
          is_executable = false;
          break;
        }
      }

      if (is_executable) {
        candidate_indices.push_back(i);
      }
    }
  }
  return candidate_indices;
}

std::pair<int, int64_t>
ShortestExpectedLatencyPlanner::GetShortestSubgraphIndex(
    std::vector<int> subgraph_indices, int64_t start_time,
    std::map<TfLiteDeviceFlags, int64_t>& device_waiting) {
  int64_t min_latency = INT_MAX;
  int min_idx = 0;

  for (auto subgraph_index : subgraph_indices) {
    SubgraphKey& key = interpreter_->subgraph(subgraph_index)->GetKey();

    int64_t waiting_time = device_waiting[key.device_flag];
    int64_t profiled = interpreter_->GetSubgraphProfileResult(key);
    int64_t expected_latency = profiled + std::max(waiting_time, start_time);

    if (min_latency > expected_latency) {
      min_latency = expected_latency;
      min_idx = subgraph_index;
    }
  }
  return {min_idx, min_latency};
}

}  // namespace impl
}  // namespace tflite
