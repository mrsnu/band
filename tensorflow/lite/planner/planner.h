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

  void Plan();

  // Check whether profiling is required or not.
  bool NeedProfile();

  // Enqueues a job to a worker request queue.
  int EnqueueRequest(Job job, bool push_front = false);

  // Enqueues a batch of jobs to a worker request queue.
  // Assigns new job id for non-continuous job.
  std::vector<int> EnqueueBatch(std::vector<Job> jobs, bool push_front = false);

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

  int GetWindowSize() const { return schedule_window_size_; }

  void SetWindowSize(int schedule_window_size);

  const std::map<int, int>& GetModelExecutionCounts() const {
    return model_execution_count_;
  }

  // Get the Job instance with the `job_id`.
  Job GetFinishedJob(int job_id);

  // Get which worker types the schedulers require.
  int GetWorkerType() const;

  // Checks if the schedulers can handle fallback subgraphs.
  // Returns true if any of the scheduler can handle fallback subgraphs.
  // But, note that having both types of scheduler (w/ fallback, w/o fallback),
  // may lead to unexpected results.
  bool NeedFallbackSubgraphs() const;

  // Write job logs and delete the job from the finished queue.
  void FlushFinishedJobs();

  // Copy the Job instances from the `requests_` to the local queue.
  // Note that this function is to minimize the hold time for the queue lock.
  void CopyToLocalQueue(JobQueue& local_jobs);

  // Enqueue the request to the worker.
  void EnqueueToWorkers(ScheduleAction& action);

  // Check if the job violated the specified SLO.
  // This func assumes that workers_waiting_, job.profiled_time,
  // job.device_id, and job.enqueue_time are all up to date.
  bool IsSLOViolated(Job& job);

  // Set the job status and enqueue to the finished queue.
  void HandleSLOViolatedJob(Job& job);

  // Update the current device waiting time.
  void UpdateWorkerWaitingTime();

  // Update `model_worker_map_`.
  void TryUpdateModelWorkerMapping();

  // Get idle workers from `workers_waiting_`.
  // NOTE: Another option to implement the function is to be pass
  // the current WorkerWaitingTime as a parameter.
  std::set<int> GetIdleWorkers();

  WorkerWaitingTime& GetWorkerWaitingTime() { return workers_waiting_; }

  int IssueSchedId();

  std::map<int, int>& GetModelWorkerMap() { return model_worker_map_; }

  void UpdateJobScheduleStatus(Job& job, Subgraph* target_subgraph);

  void PrepareReenqueue(Job& job);

 private:
  bool IsJobIdValid(int job_id);
  int GetJobRecordIndex(int job_id) const;

  CpuSet cpu_set_;
  bool need_cpu_update_ = false;

  SafeBool planner_safe_bool_;

  // Jobs Finished
  ConcurrentJobQueue jobs_finished_;
  std::map<int, int> model_execution_count_;

  // Request Queue
  ConcurrentJobQueue requests_;

  // Multi-level Local Queue.
  // The closer the index is to 0, the higher the priority.
  std::vector<JobQueue> local_queues_;
  std::vector<std::unique_ptr<Scheduler>> schedulers_;

  std::array<Job, NUM_FINISHED_RECORDS> jobs_finished_record_;
  int num_submitted_jobs_ = 0;
  int num_finished_jobs_ = 0;

  std::condition_variable end_invoke_;
  std::string log_path_;

  int schedule_window_size_ = INT_MAX;

  std::thread planner_thread_;
  int sched_id_ = 0;
  WorkerWaitingTime workers_waiting_;
  // Map structure to find assigned device of model idx (model_id, device flag)
  std::map<int, int> model_worker_map_;
  Interpreter* interpreter_;
};

class Scheduler {
 public:
  explicit Scheduler(Planner* planner) : planner_(planner) {}
  // A Schedule() function is expected to do the followings:
  // For the given requests, selected requests to schedule and
  // find the appropriate devices. The selected requests should be
  // enqueued in the `action_`.
  virtual void Schedule(JobQueue& requests) = 0;
  Interpreter* GetInterpreter() { return planner_->GetInterpreter(); }
  int IssueSchedId() { return planner_->IssueSchedId(); }
  WorkerWaitingTime& GetWorkerWaitingTime() {
    return planner_->GetWorkerWaitingTime();
  }
  bool NeedProfile() { return need_profile_; }
  bool NeedFallbackSubgraphs() { return need_fallback_subgraphs_; }
  TfLiteWorkerType GetWorkerType() { return worker_type_; }
  ScheduleAction& GetAction() { return action_; }
  void EnqueueAction(Job job, Subgraph* subgraph);

 protected:
  bool need_profile_;
  bool need_fallback_subgraphs_;
  TfLiteWorkerType worker_type_;
  Planner* planner_;
  ScheduleAction action_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_PLANNER_H_
