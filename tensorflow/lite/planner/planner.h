#ifndef TENSORFLOW_LITE_PLANNER_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_PLANNER_H_

#include <memory>
#include <list>
#include <string>
#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/safe_bool.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

namespace impl {

class Interpreter;

// Contains how a Subgraph should be executed.
// Currently, the unit of device placement is a `Subgraph`.
// Each Subgraph contains one `ModelPlan` as a member.
struct ModelPlan {
 public:
  ModelPlan():device_(kTfLiteCPU) {}
  ModelPlan(ModelPlan&&) = default;
  ModelPlan(const ModelPlan&) = delete;
  TfLiteDeviceFlags device_;  
};

// assigns requested model to devices according to `ModelPlan` of a `Subgraph`.
// The interpreter manages a `Planner`.
class Planner {
 public:
  explicit Planner(Interpreter* interpreter);
  ~Planner();

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
  void EnqueueRequest(Job job);

  // Enqueues a batch of jobs to a worker request queue.
  void EnqueueBatch(std::list<Job> jobs);

  // Waits until the jobs are done.
  // The interpreter calls the method.
  // TODO #18: Make the planner run in a different thread
  TfLiteStatus Wait(int num_requests);

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
    return requests_mtx_;
  }

  std::deque<Job>& GetRequests() {
    return requests_;
  }

 protected:
  std::thread planner_thread_;

 private:
  Interpreter* interpreter_;
  SafeBool planner_safe_bool_;

  // Jobs Finished
  std::mutex job_queue_mtx_;
  std::deque<Job> jobs_finished_;

  // Request Queue
  std::mutex requests_mtx_;
  std::deque<Job> requests_;

  std::condition_variable end_invoke_;

  // TODO #36: Make this a configurable option (command line arg)
  std::string log_path_ = "/data/local/tmp/model_execution_log.csv";
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_PLANNER_H_
