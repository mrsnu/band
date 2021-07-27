#ifndef TENSORFLOW_LITE_PLANNER_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_PLANNER_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/planner/util.h"
#include "tensorflow/lite/safe_bool.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/worker.h"

namespace tflite {

namespace impl {

class Interpreter;
class Scheduler;

// The interpreter manages a `Planner`.
class Planner {
 public:
  explicit Planner(Interpreter* interpreter);
  ~Planner();

  TfLiteStatus Init(PlannerConfig& config);

  /*
  Derived classes should generally follow this template when implementing
  `Plan()`: while (true) {
          // sleep until somebody wakes me up with GetSafeBool().notify()
          if (GetSafeBool().wait()) return;
          // wake up and do something with the request queue
          std::unique_lock<std::mutex> lock(GetRequestsMtx()); // exclusive
  access to the request queue Job j = GetRequests().front(); // get the first
  job GetRequests().pop_front(); // actual dequeue
          // enqueue the job in the correct worker queue
          // Worker& worker = GetInterpreter()->GetWorker(device_idx);
          // ...
  }
  */
  virtual void Plan();

  // Check whether profiling is required or not.
  bool NeedProfile();

  // Enqueues a job to a worker request queue.
  int EnqueueRequest(Job job);

  // Enqueues a batch of jobs to a worker request queue.
  // Assigns new job id for non-continuous job.
  std::vector<int> EnqueueBatch(std::vector<Job> jobs);

  // Waits until the jobs are done.
  // The interpreter calls the method.
  void Wait(std::vector<int> job_ids);
  void WaitAll();

  // Enqueues a finised job to the queue.
  // A worker calls the method.
  // TODO #18: Make the planner run in a different thread
  void EnqueueFinishedJob(Job job);

  Interpreter* GetInterpreter() { return interpreter_; }

  SafeBool& GetSafeBool() { return planner_safe_bool_; }

  std::mutex& GetRequestsMtx() { return requests_.mtx; }

  JobQueue& GetRequests() { return requests_.queue; }

  int GetWindowSize() { return schedule_window_size_; }

  void SetWindowSize(int schedule_window_size);

  const std::map<int, int>& GetModelExecutionCounts() const {
    return model_execution_count_;
  }

  // Get the Job instance with the `job_id`.
  Job GetFinishedJob(int job_id);

  // Get which worker types the scheduers require.
  int GetWorkerType();

  // Checks if the schedulers can handle fallback subgraphs.
  // Returns true if any of the scheduler can handle fallback subgraphs.
  bool RequireFallbackSubgraphs();

  // Write job logs and delete the job from the finished queue.
  void FlushFinishedJobs();

  // Copy the Job instances from the `requests_` to the local queue.
  // Note that this function is to minimize the hold time for the queue lock.
  void CopyToLocalQueue(JobQueue& local_jobs);

  // Enqueue the request to the worker.
  void EnqueueToWorkers(ScheduleAction& action);

  // Check if the job violated the specified SLO.
  void CheckSLOViolation(Job& job);

  // Update the current device waiting time.
  void UpdateDeviceWaitingTime();

  // Update `model_device_map_`.
  void UpdateModelDeviceMapping();

  // Get idle devices from `device_waiting_`.
  std::set<TfLiteDeviceFlags> GetIdleDevices();

  DeviceWaitingTime& GetDeviceWaitingTime() { return device_waiting_; }

  int IssueSchedId() { return sched_id_++; }

  std::map<int, int>& GetModelDeviceMap() { return model_device_map_; }

 private:
  bool IsJobIdValid(int job_id);
  int GetJobRecordIndex(int job_id) const;

  Interpreter* interpreter_;
  SafeBool planner_safe_bool_;

  // Jobs Finished
  ConcurrentJobQueue jobs_finished_;
  std::map<int, int> model_execution_count_;

  // Request Queue
  ConcurrentJobQueue requests_;

  // Multi-level Local Queue.
  // The the index is closer to 0, the higher the priority.
  std::vector<JobQueue> local_queues_;
  std::map<int, std::unique_ptr<Scheduler>> schedulers_;

  std::array<Job, NUM_FINISHED_RECORDS> jobs_finished_record_;
  int num_submitted_jobs_ = 0;
  int num_finished_jobs_ = 0;

  std::condition_variable end_invoke_;
  std::string log_path_;

  int schedule_window_size_ = INT_MAX;

  std::thread planner_thread_;
  int sched_id_ = 0;
  DeviceWaitingTime device_waiting_;
  // Map structure to find assigned device of model idx (model_id, device flag)
  std::map<int, int> model_device_map_;
};

class Scheduler {
 public:
  explicit Scheduler(Planner* planner) : planner_(planner) {}
  virtual ScheduleAction Schedule(JobQueue& requests) = 0;
  Interpreter* GetInterpreter() { return planner_->GetInterpreter(); }
  int IssueSchedId() { return planner_->IssueSchedId(); }
  DeviceWaitingTime& GetDeviceWaitingTime() {
    return planner_->GetDeviceWaitingTime();
  }
  bool NeedProfile() { return need_profile_; }
  bool NeedFallbackSubgraphs() { return need_fallback_subgraphs_; }
  WorkerType GetWorkerType() { return worker_type_; }

 protected:
  bool need_profile_;
  bool need_fallback_subgraphs_;
  WorkerType worker_type_;
  Planner* planner_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_PLANNER_H_
