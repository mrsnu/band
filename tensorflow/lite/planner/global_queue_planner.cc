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
    std::map<TfLiteDeviceFlags, int64_t> device_waiting;
    for (int i = 0; i < kTfLiteNumDevices; ++i) {
      TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
      Worker* worker = GetInterpreter()->GetWorker(device_flag);
      if (worker != nullptr) {
        if (!worker->IsBusy()) {
        idle_devices.insert(device_flag);
        }
        device_waiting[device_flag] = worker->GetWaitingTime();
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
    while (!idle_devices.empty() && it != ordered_requests_.end()) {
      if (std::distance(ordered_requests_.begin(), it) > GetWindowSize()) {
        // std::cout << "Limited by window size. " << std::endl;
        break;
      }

      std::pair<int, int64_t> best_subgraph =
          GetInterpreter()->GetShortestLatency(it->model_id,
                                               it->start_idx,
                                               0,
                                               device_waiting,
                                               available_devices);

      int subgraph_idx = best_subgraph.first;
      int64_t expected_end_time = best_subgraph.second;

      if (subgraph_idx == -1) {
        // no device can run me at the moment
        // just wait...
        ++it;
        continue;
      }

      Job job = *it;

      /* Validate Queue Order
      Job first_job = *(ordered_requests_.begin());
      for (auto test_it = ordered_requests_.begin();
           test_it != ordered_requests_.end();
           ++test_it) {
        double first_deadline = first_job.enqueue_time + first_job.slo;
        double current_deadline = (*test_it).enqueue_time + (*test_it).slo;

        if (first_deadline > current_deadline) {
          std::cout << "Queue Order is not correct." << std::endl;
        }
      }*/

      int64_t current_time = profiling::time::NowMicros();
      if (current_time + expected_end_time > it->enqueue_time + it->slo) {
        // SLO violation!! drop!
        assert(job.is_finished);
        job.end_time = LLONG_MAX;
        EnqueueFinishedJob(job);
        it = ordered_requests_.erase(it);
        continue;
      }

      SubgraphKey& key = GetInterpreter()->subgraph(subgraph_idx)->GetKey();
      int64_t profile_result =
          GetInterpreter()->GetSubgraphProfileResult(key);
      if (busy_devices.find(key.device_flag) != busy_devices.end()) {
        // Selected device is busy.
        // just wait...
        // but make sure to increase device waiting time
        // assuming the subgraph is assigned to the best device.
        ++it;
        device_waiting[key.device_flag] += profile_result;
        /*
        std::cout << "Selected Busy Device: " << key.device_flag
                  << ", waiting time : " << device_waiting[key.device_flag]
                  << std::endl;
        */
        continue;
      }
      job.start_idx = key.start_idx;
      job.end_idx = key.end_idx;
      job.subgraph_idx = subgraph_idx;
      job.device_id = key.device_flag;
      job.sched_id = sched_id++;

      job.expected_execution_time_us = profile_result;
      if (job.expected_latency_us == 0) {
        job.expected_latency_us = expected_end_time;
      }

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
  if (job.enqueue_time == 0) {
    job.enqueue_time = profiling::time::NowMicros();
  }
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
    if (job.enqueue_time == 0) {
      job.enqueue_time = enqueue_time;
    }
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
