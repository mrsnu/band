#ifndef BAND_JOB_H_
#define BAND_JOB_H_

#include "band/common.h"

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
    kEnqueueFailed,
    kInputCopyFailure,
    kOutputCopyFailure,
    kInvokeFailure,
  };

  explicit Job() {}
  explicit Job(ModelId model_id) : model_id(model_id) {}
  explicit Job(ModelId model_id, int64_t slo)
      : model_id(model_id), slo_us(slo) {}

  bool IsInitialized() const;
  std::string ToJson() const;

  // Constant variables (Valid after invoke)
  // TODO: better job life-cycle to change these to `const`
  ModelId model_id = -1;;
  int input_handle = -1;
  int output_handle = -1;
  JobId job_id = -1;
  bool require_callback = true;

  // For record (Valid after execution)
  int64_t enqueue_time = 0;
  int64_t invoke_time = 0;
  int64_t end_time = 0;
  // Profiled invoke execution time
  int64_t profiled_execution_time = 0;
  // Expected invoke execution time
  int64_t expected_execution_time = 0;
  // Expected total latency
  int64_t expected_latency = 0;
  int64_t slo_us;

  // Target worker id (only for fixed worker request)
  WorkerId target_worker_id = -1;

  // Current status for execution (Valid after planning)
  Status status = Status::kNone;
  SubgraphKey subgraph_key;
  std::vector<Job> following_jobs;

  // Resolved unit subgraphs and executed subgraph keys
  BitMask resolved_unit_subgraphs;
  std::list<SubgraphKey> previous_subgraph_keys;
 private:
  
};

}  // namespace band

#endif  // BAND_JOB_H_