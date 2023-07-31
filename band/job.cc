#include "band/job.h"
#include "absl/strings/str_format.h"
#include "band/logger.h"

namespace band {

std::size_t JobIdBitMaskHash::operator()(
    const std::pair<int, BitMask>& p) const {
  auto hash_func = std::hash<int>();
  return hash_func(p.first) ^ hash_func(p.second.to_ullong());
}

bool Job::IsEnqueued() const {
  return (model_id_ != -1) && (id_ != -1) && (input_handle_ != -1) &&
         (output_handle_ != -1);
}

absl::Status Job::Enqueue(JobId id) {
  if (status_ != Status::kNone) {
    return absl::InternalError("Job is already enqueued");
  }
  status_ = Status::kQueued;
  id_ = id;
  return absl::OkStatus();
}

absl::Status Job::Success() {
  status_ = Status::kSuccess;
  return absl::OkStatus();
}

absl::Status Job::SLOViolation() {
  status_ = Status::kSLOViolation;
  return absl::OkStatus();
}

absl::Status Job::EnqueueFailure() {
  status_ = Status::kEnqueueFailure;
  return absl::OkStatus();
}

absl::Status Job::InputCopyFailure() {
  status_ = Status::kInputCopyFailure;
  return absl::OkStatus();
}

absl::Status Job::OutputCopyFailure() {
  status_ = Status::kOutputCopyFailure;
  return absl::OkStatus();
}

absl::Status Job::InvokeFailure() {
  status_ = Status::kInvokeFailure;
  return absl::OkStatus();
}

Job Job::Next(const SubgraphKey& target_key, int profiled_execution_time,
              int expected_execution_time, bool last) {
  Job job = *this;
  job.subgraph_key = target_key;
  job.profiled_execution_time = profiled_execution_time;
  job.expected_execution_time = expected_execution_time;
  job.resolved_unit_subgraphs |= target_key.GetUnitIndices();

  if (!last) {
    Job remaining_ops;
    remaining_ops.id_ = id_;
    remaining_ops.model_id_ = model_id_;
    remaining_ops.input_handle_ = input_handle_;
    remaining_ops.output_handle_ = output_handle_;
    remaining_ops.slo_us_ = slo_us_;

    remaining_ops.enqueue_time = enqueue_time;
    remaining_ops.expected_latency = expected_latency;
    remaining_ops.following_jobs = following_jobs;
    remaining_ops.resolved_unit_subgraphs = resolved_unit_subgraphs;
    remaining_ops.previous_subgraph_keys = previous_subgraph_keys;
    remaining_ops.previous_subgraph_keys.push_back(subgraph_key);

    job.following_jobs.clear();
    job.following_jobs.push_back(remaining_ops);
  }

  return job;
}

std::string Job::ToJson() const {
  return "{\"enqueue_time\":" + std::to_string(enqueue_time) +
         ",\"invoke_time\":" + std::to_string(invoke_time) +
         ",\"end_time\":" + std::to_string(end_time) +
         ",\"profiled_execution_time\":" +
         std::to_string(profiled_execution_time) +
         ",\"expected_execution_time\":" +
         std::to_string(expected_execution_time) +
         ",\"expected_latency\":" + std::to_string(expected_latency) +
         ",\"slo_us\":" + std::to_string(slo_us_) +
         ",\"model_id\":" + std::to_string(model_id_) +
         ",\"unit_indices\":" + subgraph_key.GetUnitIndicesString() +
         ",\"job_id\":" + std::to_string(id_) + "}";
}

std::string Job::ToString() const {
  return absl::StrFormat(
      "Job %d (Model %d), Subgraph: %s, input_handle: %d, output_handle: %d",
      id_, model_id_, subgraph_key.ToString(), input_handle_, output_handle_);
}

}  // namespace band