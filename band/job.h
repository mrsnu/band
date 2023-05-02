#ifndef BAND_JOB_H_
#define BAND_JOB_H_

#include "band/common.h"
#include "band/time.h"

#include "absl/status/status.h"
#include "absl/types/optional.h"

#define REPORT_IF_JOB_METHOD_FAILS(method)                                     \
  do {                                                                         \
    absl::Status status = (method);                                            \
    if (!(status.ok())) {                                                      \
      BAND_REPORT_ERROR(GetErrorReporter(), "Failed to update job status: %s", \
                        status.message());                                     \
    }                                                                          \
  } while (0);

namespace band {

class Context;

struct JobStatus {
 public:
  enum class State : size_t { Created = 0, Queued, Invoked, Success, Error };
  enum class ErrorState : size_t {
    SLOViolation = 0,
    FailInvoke = 1,
    FailCopyInput = 2,
  };
  State state;
  absl::optional<ErrorState> error_state;
  std::string error_message;

  static JobStatus Created() {
    return JobStatus{.state = State::Created,
                     .error_state = absl::nullopt,
                     .error_message = ""};
  }

  static JobStatus Queued() {
    return JobStatus{.state = State::Queued,
                     .error_state = absl::nullopt,
                     .error_message = ""};
  }

  static JobStatus Invoked() {
    return JobStatus{.state = State::Invoked,
                     .error_state = absl::nullopt,
                     .error_message = ""};
  }

  static JobStatus Success() {
    return JobStatus{.state = State::Success,
                     .error_state = absl::nullopt,
                     .error_message = ""};
  }

  static JobStatus Error(ErrorState error_state, std::string error_message) {
    return JobStatus{.state = State::Error,
                     .error_state = error_state,
                     .error_message = error_message};
  }
};
std::string GetName(JobStatus job_status);
std::ostream& operator<<(std::ostream& os, const JobStatus& status);

class Job {
 public:
  Job() = default;
  Job(ModelId model_id, int input_handle, int output_handle,
      absl::optional<int64_t> slo_us, bool require_callback = true)
      : model_id_(model_id),
        input_handle_(input_handle),
        output_handle_(output_handle),
        slo_us_(slo_us),
        require_callback_(require_callback) {}

  void SetTargetWorker(WorkerId target_worker_id) {
    target_worker_id_ = target_worker_id;
  }
  void SetSloUs(int64_t slo_us) { slo_us_ = slo_us; }
  int GetInputHandle() const { return input_handle_; }
  int GetOutputHandle() const { return output_handle_; }

  // State
  absl::Status Queued(int32_t id);
  absl::Status Invoked();
  absl::Status Success();
  void Error(JobStatus::ErrorState error_state, std::string message);

  // Update information in `Queued` state
  absl::Status AssignSubgraphKey(SubgraphKey key);
  absl::Status AssignSchedId(int sched_id);
  absl::Status UpdateProfileInfo(int64_t profiled_execution_time,
                                 int64_t expected_execution_time);
  absl::Status UpdateSubgraphSchedule(Context& context);
  absl::Status UpdateExpectedLatency(int64_t expected_latency);
  void PrepareReenqueue() {
    invoke_time_ = 0;
    end_time_ = 0;
    resolved_unit_subgraphs_ = 0;
    following_jobs_.clear();
  }

  // Update information in `Invoked` state
  absl::Status UpdateEndTime();

  // State getter
  JobStatus status() const { return status_; }
  bool IsCreated() const { return status_.state == JobStatus::State::Created; }
  bool IsQueued() const { return status_.state == JobStatus::State::Queued; }
  bool IsInvoked() const { return status_.state == JobStatus::State::Invoked; }
  bool IsSuccess() const { return status_.state == JobStatus::State::Success; }
  bool IsError() const { return status_.state == JobStatus::State::Error; }
  JobStatus::ErrorState GetErrorState() const {
    return status_.error_state.value();
  }
  std::string GetErrorMessage() const { return status_.error_message; }

  // Check
  bool HasTargetWorkerId() const { return target_worker_id_.has_value(); }

  // Must be called after the `Created` state
  ModelId model_id() const { return model_id_; }
  int input_handle() const { return input_handle_; }
  int output_handle() const { return output_handle_; }
  int64_t slo_us() const { return slo_us_.value(); }
  bool require_callback() const { return require_callback_; }
  WorkerId target_worker_id() const { return target_worker_id_.value(); }

  // Must be called after the `Queued` state
  int32_t id() const { return id_.value(); }
  SubgraphKey subgraph_key() const { return subgraph_key_.value(); }
  int32_t sched_id() const { return sched_id_.value(); }
  int64_t enqueue_time() const { return enqueue_time_.value(); }
  int64_t profiled_execution_time() const {
    return profiled_execution_time_.value();
  }
  int64_t expected_execution_time() const {
    return expected_execution_time_.value();
  }
  int64_t expected_latency() const { return expected_latency_.value(); }
  std::list<SubgraphKey> previous_subgraph_keys() const {
    return previous_subgraph_keys_;
  }
  BitMask resolved_unit_subgraphs() const { return resolved_unit_subgraphs_; }
  std::vector<Job> following_jobs() const { return following_jobs_; }

  // Must be called after the `Invoked` state
  int64_t invoke_time() const { return invoke_time_.value(); }
  int64_t end_time() const { return end_time_.value(); }

  std::string ToJson() const;

 private:
  JobStatus status_ = JobStatus::Created();

  // Arbitrary job creation is not allowed from outside.
  Job(JobStatus status, ModelId model_id, int input_handle, int output_handle,
      absl::optional<int64_t> slo_us, bool require_callback,
      absl::optional<WorkerId> target_worker_id, absl::optional<int32_t> id,
      absl::optional<SubgraphKey> subgraph_key,
      absl::optional<int32_t> sched_id, absl::optional<int64_t> enqueue_time,
      absl::optional<int64_t> profiled_execution_time,
      absl::optional<int64_t> expected_execution_time,
      absl::optional<int64_t> expected_latency,
      std::list<SubgraphKey> previous_subgraph_keys,
      BitMask resolved_unit_subgraphs, std::vector<Job> following_jobs)
      : status_(status),
        model_id_(model_id),
        input_handle_(input_handle),
        output_handle_(output_handle),
        slo_us_(slo_us),
        require_callback_(require_callback),
        target_worker_id_(target_worker_id),
        id_(id),
        subgraph_key_(subgraph_key),
        sched_id_(sched_id),
        enqueue_time_(enqueue_time),
        profiled_execution_time_(profiled_execution_time),
        expected_execution_time_(expected_execution_time),
        expected_latency_(expected_latency),
        previous_subgraph_keys_(previous_subgraph_keys),
        resolved_unit_subgraphs_(resolved_unit_subgraphs),
        following_jobs_(following_jobs){};

  // Must be initialized in the `Created` state
  ModelId model_id_;
  int input_handle_;
  int output_handle_;
  absl::optional<int64_t> slo_us_ = absl::nullopt;
  bool require_callback_ = true;
  absl::optional<WorkerId> target_worker_id_ = absl::nullopt;

  // Must be initialized in the `Queued` state;
  absl::optional<int32_t> id_ = absl::nullopt;
  absl::optional<SubgraphKey> subgraph_key_ = absl::nullopt;
  absl::optional<int32_t> sched_id_ = absl::nullopt;
  absl::optional<int64_t> enqueue_time_ = absl::nullopt;
  absl::optional<int64_t> profiled_execution_time_ = absl::nullopt;
  absl::optional<int64_t> expected_execution_time_ = absl::nullopt;
  absl::optional<int64_t> expected_latency_ = absl::nullopt;
  std::list<SubgraphKey> previous_subgraph_keys_;
  BitMask resolved_unit_subgraphs_;
  std::vector<Job> following_jobs_;

  // Must be initialized in the `Invoked` state;
  absl::optional<int64_t> invoke_time_ = absl::nullopt;
  absl::optional<int64_t> end_time_ = absl::nullopt;
};

}  // namespace band

#endif  // BAND_JOB_H_