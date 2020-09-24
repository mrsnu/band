#include "tensorflow/lite/planner.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace impl {

Planner::Planner(Interpreter* interpreter) {
  interpreter_ = interpreter;
  for (int i = 0; i < interpreter_->GetNumDevices(); ++i) {
    workers_.emplace_back(new Worker(interpreter, this));
  }
}

Planner::~Planner() {
  kill_workers_ = true;
}

TfLiteStatus Planner::Plan() {
  if (!change_plan_) return kTfLiteOk;
  change_plan_ = false;
  TfLiteStatus status;

  for (int i = 0; i < interpreter_->subgraphs_size(); ++i) {
    Subgraph& subgraph = *(interpreter_->subgraph(i));
    status = subgraph.UndoAllDelegates();
    if (status != kTfLiteOk)
      return status;

    if (i % kTfLiteNumDevices == kTfLiteGPU) {
      subgraph.GetModelPlan()->device_ = kTfLiteGPU;
    } else if (i % kTfLiteNumDevices == kTfLiteDSP) {
      subgraph.GetModelPlan()->device_ = kTfLiteDSP;
    }

    if (subgraph.GetModelPlan()->device_ != kTfLiteCPU) {
      status = subgraph.ModifyGraphWithDelegate(
          interpreter_->device_delegates(subgraph.GetModelPlan()->device_));
    }
    if (status != kTfLiteOk)
      return status;
  }

  return kTfLiteOk;
}


TfLiteStatus Planner::Wait(int num_requests) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  end_invoke_.wait(lock, [this, num_requests]{
    return jobs_finished_.size() >= num_requests;
  });

  for (int i = 0; i < num_requests; ++i) {
    jobs_finished_.pop_front();
  }
  lock.unlock();

  return kTfLiteOk;
}

void Planner::EnqueueFinishedJob(Job job) {
  std::unique_lock<std::mutex> lock(job_queue_mtx_);
  jobs_finished_.push_back(job);
  lock.unlock();

  end_invoke_.notify_one();
}

void Planner::EnqueueRequest(TfLiteDevice device_idx, Job job) {
  Worker& worker = *workers_[device_idx];

  std::unique_lock<std::mutex> lock(worker.device_mtx_);
  worker.requests_.push_back(job);
  lock.unlock();

  worker.request_cv_.notify_one();
}

}  // namespace impl
}  // namespace tflite
