#ifndef TENSORFLOW_LITE_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_H_

#include <memory>
#include <fstream>
#include <list>

#include "tensorflow/lite/worker.h"
#include "tensorflow/lite/safe_bool.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

namespace impl {

class Interpreter;

typedef enum {
  kTfLiteCPU = 0,
  kTfLiteGPU = 1,
  kTfLiteDSP = 2,
  kTfLiteTPU = 3,
  kTfLiteNumDevices = 4,
} TfLiteDevice;

// Contains how a Subgraph should be executed.
// Currently, the unit of device placement is a `Subgraph`.
// Each Subgraph contains one `ModelPlan` as a member.
struct ModelPlan{
 public:
  ModelPlan():device_(kTfLiteCPU) {}
  ModelPlan(ModelPlan&&) = default;
  ModelPlan(const ModelPlan&) = delete;
  TfLiteDevice device_;

  // Flag from interpreter_builder
  // Use MaybeCreateXNNPACKDelegate(num_threads); to create a XNN delegate
  bool can_use_xnn_pack_ = false;
  // TODO: Move AcquireFlexDelegate(); in interpreter_builder.cc  
  // somewhere to create / use a flex delegate
  bool has_flex_op_ = false;
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

  // Enqueues a job to a worker request queue.
  void EnqueueRequest(Job job);

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
  std::thread planner_thread_;

  std::ofstream log_file_;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_H_
