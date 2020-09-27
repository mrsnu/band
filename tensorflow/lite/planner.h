#ifndef TENSORFLOW_LITE_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_H_

#include <memory>
#include "tensorflow/lite/worker.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

namespace tflite {

namespace impl {

class Interpreter;
class Subgraph;

typedef enum {
  kTfLiteCPU = 0,
  kTfLiteGPU = 1,
  kTfLiteDSP = 2,
  kTfLiteNumDevices = 3,
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
  ~Planner() = default;

  TfLiteStatus Plan();

  // Enqueues a job to a worker request queue.
  void EnqueueRequest(Job job);

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

 private:
  Interpreter* interpreter_;

  std::mutex job_queue_mtx_;
  std::deque<Job> jobs_finished_;
  std::condition_variable end_invoke_;

  bool change_plan_ = true;
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_H_
