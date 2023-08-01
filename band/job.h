#ifndef BAND_JOB_H_
#define BAND_JOB_H_

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "band/common.h"
#include "band/estimator/record.h"

typedef int JobId;

namespace band {

// hash function to use pair<int, BitMask> as map key in cache_
// https://stackoverflow.com/a/32685618
struct JobIdBitMaskHash {
  std::size_t operator()(const std::pair<int, BitMask>& p) const;
};

// Job struct is the scheduling and executing unit.
// The request can specify a model by indication the model id
class Job {
 public:
  enum class Status : size_t {
    kNone,
    kQueued,
    kSuccess,
    kFinished,
    kSLOViolation,
    kEnqueueFailure,
    kInputCopyFailure,
    kOutputCopyFailure,
    kInvokeFailure,
  };

  explicit Job() {}
  explicit Job(ModelId model_id, int input_handle, int output_handle,
               int64_t slo_us = -1)
      : model_id_(model_id),
        input_handle_(input_handle),
        output_handle_(output_handle),
        slo_us_(slo_us) {}

  Status status() const { return status_; }
  ModelId model_id() const { return model_id_; }
  int input_handle() const { return input_handle_; }
  int output_handle() const { return output_handle_; }
  JobId id() const { return id_; }
  int64_t slo_us() const { return slo_us_; }
  bool require_callback() const { return require_callback_; }
  WorkerId target_worker_id() const { return target_worker_id_; }

  bool HasSLO() const { return slo_us_ > 0; };
  bool HasTargetWorker() const { return target_worker_id_ != -1; };

  bool IsEnqueued() const;

  // Methods for job construction.
  absl::Status RequireCallback() {
    if (status_ != Status::kNone) {
      return absl::InternalError("Job is already enqueued");
    }
    require_callback_ = true;
    return absl::OkStatus();
  };

  absl::Status SetTargetWorker(WorkerId target_worker_id) {
    if (status_ != Status::kNone) {
      return absl::InternalError("Job is already enqueued");
    }
    target_worker_id_ = target_worker_id;
    return absl::OkStatus();
  }

  // Normal states
  absl::Status Enqueue(JobId id);

  absl::Status Success();

  // Failure states
  absl::Status SLOViolation();
  absl::Status EnqueueFailure();
  absl::Status InputCopyFailure();
  absl::Status OutputCopyFailure();
  absl::Status InvokeFailure();

  Job Next(const SubgraphKey& target_key,
           absl::optional<LatencyRecord> latency_profile, bool last);

  std::string ToJson() const;
  std::string ToString() const;

  // Constant variables (Valid after invoke)

  // For record (Valid after execution)
  int64_t enqueue_time = 0;
  int64_t invoke_time = 0;
  int64_t end_time = 0;
  // Profiled invoke execution time
  absl::optional<LatencyRecord> latency_profile;
  int64_t profiled_execution_time = 0;
  int64_t expected_execution_time = 0;
  // Expected total latency
  int64_t expected_latency = 0;

  // Current status for execution (Valid after planning)
  SubgraphKey subgraph_key;
  std::vector<Job> following_jobs;

  // Resolved unit subgraphs and executed subgraph keys
  BitMask resolved_unit_subgraphs = 0;
  std::list<SubgraphKey> previous_subgraph_keys;

 private:
  Status status_ = Status::kNone;

  JobId id_ = -1;
  ModelId model_id_ = -1;
  int input_handle_ = -1;
  int output_handle_ = -1;
  int64_t slo_us_ = -1;
  bool require_callback_ = false;

  // Target worker id (only for fixed worker request)
  WorkerId target_worker_id_ = -1;
};

}  // namespace band

#endif  // BAND_JOB_H_