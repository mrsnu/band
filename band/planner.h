#ifndef BAND_PLANNER_H_
#define BAND_PLANNER_H_

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "band/config.h"
#include "band/safe_bool.h"
#include "band/scheduler/scheduler.h"
#include "band/worker.h"

#include "absl/status/statusor.h"

namespace band {

// The maximum number of available job outputs at one time.
#define NUM_FINISHED_RECORDS 1000

// The job queue which can be shared by multiple threads.
struct ConcurrentJobQueue {
  JobQueue queue;
  std::mutex mtx;
};

class Planner {
 public:
  explicit Planner(Context& context);
  ~Planner();

  absl::Status Init(const PlannerConfig& config);
  absl::Status AddScheduler(std::unique_ptr<IScheduler> scheduler);

  // Enqueues a job to a worker request queue.
  JobId EnqueueRequest(Job job, bool push_front = false);
  // Enqueues a batch of jobs to a worker request queue.
  // Assigns new job id for non-continuous job.
  std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                  bool push_front = false);
  // Waits until the jobs are done.
  // The interpreter calls the method.
  void Wait(std::vector<int> job_ids);
  void WaitAll();
  // Enqueues a finised job to the queue.
  // A worker calls the method.
  void EnqueueFinishedJob(Job& job);
  // Enqueue the request to the worker.
  void EnqueueToWorker(const std::vector<ScheduleAction>& action);
  void Trigger() { planner_safe_bool_.notify(); }
  int IssueSchedId() { return sched_id_++; }

  // Check whether profiling is required or not.
  bool NeedProfile();
  // Checks if the schedulers can handle fallback subgraphs.
  // Returns true if any of the scheduler can handle fallback subgraphs.
  // But, note that having both types of scheduler (w/ fallback, w/o fallback),
  // may lead to unexpected results.
  bool NeedFallbackSubgraphs() const;

  std::mutex& GetRequestsMtx() { return requests_.mtx; }
  JobQueue& GetRequests() { return requests_.queue; }
  int GetWindowSize() const { return schedule_window_size_; }
  void SetWindowSize(int schedule_window_size);
  const std::map<int, int>& GetModelExecutionCounts() const {
    return model_execution_count_;
  }
  // Sets the callback function pointer to report the end of invoke.
  void SetOnEndRequest(std::function<void(int, absl::Status)> on_end_request);
  // Get the Job instance with the `job_id`.
  absl::StatusOr<Job> GetFinishedJob(int job_id);
  // Get which worker types the schedulers require.
  int GetWorkerType() const;
  std::map<ModelId, WorkerId>& GetModelWorkerMap() { return model_worker_map_; }

  const ErrorReporter* GetErrorReporter() const {
    return context_.GetErrorReporter();
  }

 private:
  // Main loop for planner_thread_
  absl::Status Plan();
  // Write job logs and delete the job from the finished queue.
  void FlushFinishedJobs();
  // Copy the Job instances from the `requests_` to the local queue.
  // Note that this function is to minimize the hold time for the queue lock.
  void CopyToLocalQueues();
  // Check if the job violated the specified SLO.
  // This func assumes that workers_waiting_, job.profiled_time,
  // job.device_id, and job.enqueue_time are all up to date.
  bool IsSLOViolated(Job& job);
  // Set the job status and enqueue to the finished queue.
  void HandleSLOViolatedJob(Job& job);
  // Update the job information based on next target key
  void UpdateJobScheduleStatus(Job& job, const SubgraphKey& target_key);
  // Update `model_worker_map_`.
  void TryUpdateModelWorkerMapping();
  bool IsJobIdValid(int job_id);
  int GetJobRecordIndex(int job_id) const;

  CpuSet cpu_set_;
  bool need_cpu_update_ = false;

  SafeBool planner_safe_bool_;

  // Jobs Finished
  std::map<int, int> model_execution_count_;

  std::function<void(int, absl::Status)> on_end_request_;

  // Request Queue
  ConcurrentJobQueue requests_;

  // Multi-level Local Queue.
  // The closer the index is to 0, the higher the priority.
  std::vector<JobQueue> local_queues_;
  std::vector<std::unique_ptr<IScheduler>> schedulers_;

  std::array<Job, NUM_FINISHED_RECORDS> jobs_finished_record_;
  std::atomic<int> num_submitted_jobs_;
  int num_finished_jobs_ = 0;

  std::condition_variable end_invoke_;
  std::string log_path_;

  int schedule_window_size_ = INT_MAX;
  int sched_id_ = 0;

  std::thread planner_thread_;
  // Map structure to find assigned worker of model idx (model_id, worker_id)
  std::map<ModelId, WorkerId> model_worker_map_;
  Context& context_;
  bool need_reschedule_ = false;
};

}  // namespace band

#endif  // BAND_PLANNER_H_
