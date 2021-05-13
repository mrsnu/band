#include "tensorflow/lite/planner/global_queue_planner.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/worker.h"


namespace tflite {
namespace impl {

void GlobalQueuePlanner::Plan() {
  int sched_id = 0;
  std::set<TfLiteDeviceFlags> available_devices;
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
    Worker* worker = GetInterpreter()->GetWorker(device_flag);
    if (worker != nullptr) {
      available_devices.insert(device_flag);
    }
  }

  while (true) {
    if (GetSafeBool().wait()) {
      return;
    }

    std::set<TfLiteDeviceFlags> idle_devices;
    for (int i = 0; i < kTfLiteNumDevices; ++i) {
      TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
      Worker* worker = GetInterpreter()->GetWorker(device_flag);
      if (worker != nullptr && !worker->IsBusy()) {
        idle_devices.insert(device_flag);
      }
    }

    if (idle_devices.empty()) {
      continue;
    }


    std::set<TfLiteDeviceFlags> busy_devices;
    std::set_difference(available_devices.begin(), available_devices.end(),
                        idle_devices.begin(), idle_devices.end(),
                        std::inserter(busy_devices,
                                      busy_devices.begin()));

    std::lock_guard<std::mutex> lock(GetRequestsMtx());
    auto it = ordered_requests_.begin();
    std::map<TfLiteDeviceFlags, int64_t> empty_map;
    while (!idle_devices.empty() && it != ordered_requests_.end()) {
      std::pair<int, int64_t> best_subgraph =
          GetInterpreter()->GetShortestLatency(it->model_id,
                                               it->start_idx,
                                               0,
                                               empty_map,
                                               busy_devices);

      int subgraph_idx = best_subgraph.first;
      int64_t expected_end_time = best_subgraph.second;

      if (subgraph_idx == -1) {
        // no device can run me at the moment
        // just wait...
        ++it;
        continue;
      }

      Job job = *it;

      if (expected_end_time > it->enqueue_time + it->slo) {
        // SLO violation!! drop!
        assert(job.is_finished);
        job.end_time = INT_MAX;
        EnqueueFinishedJob(job);
        it = ordered_requests_.erase(it);
        continue;
      }

      SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
      job.start_idx = key.start_idx;
      job.end_idx = key.end_idx;
      job.subgraph_idx = subgraph_idx;
      job.device_id = key.device_flag;
      job.sched_id = sched_id++;
      
      ModelSpec& model_spec = GetInterpreter()->GetModelSpec(job.model_id);
      if (job.end_idx < model_spec.num_ops - 1) {
        Job remaining_ops(job.model_id);
        remaining_ops.enqueue_time = job.enqueue_time;
        remaining_ops.start_idx = job.end_idx + 1;
        remaining_ops.end_idx = model_spec.num_ops - 1;
        remaining_ops.following_jobs = job.following_jobs;
        remaining_ops.request_id = job.request_id;

        job.is_finished = false;
        job.following_jobs.clear();
        job.following_jobs.push_back(remaining_ops);
      }

      Worker* worker = GetInterpreter()->GetWorker(key.device_flag);
      if (!worker->GiveJob(job)) {
        // for some reason, the worker was busy and we couldn't assign
        // this job to it
        ++it;
        continue;
      }

      it = ordered_requests_.erase(it);
      idle_devices.erase(key.device_flag);
      busy_devices.insert(key.device_flag);
    }
  }
}

void GlobalQueuePlanner::EnqueueRequest(Job job) {
  job.enqueue_time = profiling::time::NowMicros();
  if (job.request_id < 0) {
    job.request_id = total_num_jobs_++;
  }
  std::unique_lock<std::mutex> lock(requests_mtx_);
  ordered_requests_.insert(job);
  num_submitted_jobs_++;
  lock.unlock();

  planner_safe_bool_.notify();
}

void GlobalQueuePlanner::EnqueueBatch(std::vector<Job> jobs) {
  std::unique_lock<std::mutex> lock(requests_mtx_);
  auto enqueue_time = profiling::time::NowMicros();
  for (Job job : jobs) {
    job.enqueue_time = enqueue_time;
    if (job.request_id < 0) {
      job.request_id = total_num_jobs_++;
    }
    ordered_requests_.insert(job);
    num_submitted_jobs_++;
  }
  lock.unlock();

  planner_safe_bool_.notify();
}


}  // namespace impl
}  // namespace tflte
