#ifndef TENSORFLOW_LITE_PLANNER_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_PLANNER_H_

#include <memory>
#include <vector>
#include <string>

#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/safe_bool.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/planner/util.h"

namespace tflite {

namespace impl {

class Interpreter;

// The interpreter manages a `Planner`.
class Planner {
 public:
  explicit Planner(Interpreter* interpreter);
  ~Planner();

  TfLiteStatus Init(PlannerConfig& config);

	/*
	Derived classes should generally follow this template when implementing `Plan()`:
	while (true) {
		// sleep until somebody wakes me up with GetSafeBool().notify()
		if (GetSafeBool().wait()) return;
		// wake up and do something with the request queue
		std::unique_lock<std::mutex> lock(GetRequestsMtx()); // exclusive access to the request queue
		Job j = GetRequests().front(); // get the first job
		GetRequests().pop_front(); // actual dequeue
		// enqueue the job in the correct worker queue
		// Worker& worker = GetInterpreter()->GetWorker(device_idx);
		// ...
	}
	*/
  virtual void Plan() = 0;

  // Check whether profiling is required or not.
  virtual bool NeedProfile() = 0;

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

  Interpreter* GetInterpreter() {
    return interpreter_;
  }

  SafeBool& GetSafeBool() {
    return planner_safe_bool_;
  }

  std::mutex& GetRequestsMtx() {
    return requests_.mtx;
  }

  JobQueue& GetRequests() {
    return requests_.queue;
  }

  int GetWindowSize() {
    return schedule_window_size_;
  }

  void SetWindowSize(int schedule_window_size);

  bool GetLogProcessorFrequency() const { return log_processor_frequency_; }

  const std::map<int, int>& GetModelExecutionCounts() const {
    return model_execution_count_;
  }

  Job GetFinishedJob(int job_id);

 protected:
  // Write job logs and delete the job from the finished queue.
  void FlushFinishedJobs();
  // Copy the Job instances from the `requests_` to the local queue.
  JobQueue CopyToLocalQueue();
  void UpdateJobEnqueueStatus(Job& job, SubgraphKey& target) const;
  void UpdateJobWorkerStatus(Job& job, Worker* worker) const;

  std::thread planner_thread_;
  int sched_id_ = 0;
  Interpreter* interpreter_;

 private:
  bool IsJobIdValid(int job_id);
  int GetJobRecordIndex(int job_id) const;

  SafeBool planner_safe_bool_;

  // Jobs Finished
  ConcurrentJobQueue jobs_finished_;
  std::map<int, int> model_execution_count_;

  // Request Queue
  ConcurrentJobQueue requests_;
  std::array<Job, NUM_FINISHED_RECORDS> jobs_finished_record_;
  int num_submitted_jobs_ = 0;
  int num_finished_jobs_ = 0;

  std::condition_variable end_invoke_;
  std::string log_path_;
  bool log_processor_frequency_;

  int schedule_window_size_ = INT_MAX;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_PLANNER_H_
