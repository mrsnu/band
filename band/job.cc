#include "band/job.h"

#include "band/context.h"
#include "band/logger.h"

namespace band {

absl::Status Job::Invoked() {
  if (status_.state != JobStatus::State::Queued) {
    return absl::InternalError(
        "Job::Invoked() can be called only after Queued()");
  }
  status_.state = JobStatus::State::Invoked;
  invoke_time_ = time::NowMicros();
  return absl::OkStatus();
}

absl::Status Job::Queued(int32_t id) {
  if (status_.state != JobStatus::State::Created &&
      status_.state != JobStatus::State::Queued &&
      status_.state != JobStatus::State::Invoked) {
    return absl::InternalError(
        "Job::Queued() can be called only after Created()");
  }
  status_.state = JobStatus::State::Queued;
  if (!id_.has_value()) {
    id_ = id;
  }
  if (!enqueue_time_.has_value()) {
    enqueue_time_ = time::NowMicros();
  }
  return absl::OkStatus();
}

absl::Status Job::Success() {
  if (status_.state != JobStatus::State::Invoked) {
    return absl::InternalError(
        "Job::Success() can be called only after Invoked()");
  }
  status_.state = JobStatus::State::Success;
  return absl::OkStatus();
}

void Job::Error(JobStatus::ErrorState error_state,
                        std::string message) {
  status_.state = JobStatus::State::Error;
  status_.error_state = error_state;
  status_.error_message = message;
}

absl::Status Job::AssignSubgraphKey(SubgraphKey key) {
  if (status_.state != JobStatus::State::Queued) {
    return absl::InternalError(
        "Job::AssignSubgraphKey() can be called only after Queued()");
  }
  subgraph_key_ = key;
  return absl::OkStatus();
}
absl::Status Job::AssignSchedId(int sched_id) {
  if (status_.state != JobStatus::State::Queued) {
    return absl::InternalError(
        "Job::AssignSchedId() can be called only after Queued()");
  }
  sched_id_ = sched_id;
  return absl::OkStatus();
}
absl::Status Job::UpdateProfileInfo(int64_t profiled_execution_time,
                                    int64_t expected_execution_time) {
  if (status_.state != JobStatus::State::Queued) {
    return absl::InternalError(
        "Job::UpdateProfileInfo() can be called only after Queued()");
  }
  profiled_execution_time_ = profiled_execution_time;
  expected_execution_time_ = expected_execution_time;
  return absl::OkStatus();
}
absl::Status Job::UpdateSubgraphSchedule(Context& context) {
  if (status_.state != JobStatus::State::Queued) {
    return absl::InternalError(
        "Job::UpdateSubgraphSchedule() can be called only after Queued()");
  }
  resolved_unit_subgraphs_ |= subgraph_key_.value().GetUnitIndices();
  if (!context.IsEnd(subgraph_key_.value())) {
    following_jobs_.clear();
    following_jobs_.push_back(Job(
        /*status=*/JobStatus::Queued(),
        /*model_id=*/model_id_,
        /*input_handle=*/input_handle_,
        /*output_handle=*/output_handle_,
        /*slo_us=*/slo_us_,
        /*require_callback=*/require_callback_,
        /*target_worker_id=*/target_worker_id_,
        /*id=*/id_,
        /*subgraph_key=*/absl::nullopt,
        /*sched_id=*/absl::nullopt,
        /*enqueue_time=*/enqueue_time_,
        /*profiled_execution_time=*/absl::nullopt,
        /*expected_execution_time=*/absl::nullopt,
        /*expected_latency=*/expected_latency_,
        /*previous_subgraph_keys=*/previous_subgraph_keys_,
        /*resolved_unit_subgraphs=*/resolved_unit_subgraphs_,
        /*following_jobs=*/{}));
  }
  return absl::OkStatus();
}

absl::Status Job::UpdateExpectedLatency(int64_t expected_latency) {
  if (status_.state != JobStatus::State::Queued) {
    return absl::InternalError(
        "Job::UpdateExpectedLatency() can be called only after Queued()");
  }
  expected_latency_ = expected_latency;
  return absl::OkStatus();
}

absl::Status Job::UpdateEndTime() {
  if (status_.state != JobStatus::State::Invoked) {
    return absl::InternalError(
        "Job::UpdateEndTime() can be called only after Invoked()");
  }
  end_time_ = time::NowMicros();
  return absl::OkStatus();
}

std::string Job::ToJson() const {
  return "{\"enqueue_time\":" + std::to_string(enqueue_time()) +
         ",\"invoke_time\":" + std::to_string(invoke_time()) +
         ",\"end_time\":" + std::to_string(end_time()) +
         ",\"profiled_execution_time\":" +
         std::to_string(profiled_execution_time()) +
         ",\"expected_execution_time\":" +
         std::to_string(expected_execution_time()) +
         ",\"expected_latency\":" + std::to_string(expected_latency()) +
         ",\"slo_us\":" + std::to_string(slo_us()) +
         ",\"model_id\":" + std::to_string(model_id()) +
         ",\"unit_indices\":" + subgraph_key().GetUnitIndicesString() +
         ",\"job_id\":" + std::to_string(id()) + "}";
}

std::string GetName(JobStatus job_status) {
  switch (job_status.state) {
    case JobStatus::State::Created: {
      return "Created";
    } break;
    case JobStatus::State::Queued: {
      return "Queued";
    } break;
    case JobStatus::State::Invoked: {
      return "Invoked";
    } break;
    case JobStatus::State::Success: {
      return "Success";
    } break;
    default: { return "Error (" + job_status.error_message + ")"; }
  }
  BAND_LOG_PROD(BAND_LOG_ERROR, "Unknown job status: %d", job_status);
  return "Unknown job status";
}

std::ostream& operator<<(std::ostream& os, const JobStatus& status) {
  switch (status.state) {
    case JobStatus::State::Created: {
      os << "Created";
    } break;
    case JobStatus::State::Queued: {
      os << "Queued";
    } break;
    case JobStatus::State::Invoked: {
      os << "Invoked";
    } break;
    case JobStatus::State::Success: {
      os << "Success";
    } break;
    default: { os << "Error (" << status.error_message << ")"; }
  }
  BAND_LOG_PROD(BAND_LOG_ERROR, "Unknown job status: %d", status);
  return os;
}

}  // namespace band