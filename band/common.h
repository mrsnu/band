#ifndef BAND_COMMON_H_
#define BAND_COMMON_H_

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "band/c/common.h"

namespace Band {
typedef int WorkerId;
typedef int ModelId;
typedef int JobId;

// data structure for identifying subgraphs within whole models
class SubgraphKey {
 public:
  SubgraphKey();
  SubgraphKey(ModelId model_id, WorkerId worker_id,
              std::set<int> unit_indices = {});
  bool operator<(const SubgraphKey& key) const;
  bool operator==(const SubgraphKey& key) const;

  ModelId GetModelId() const { return model_id; }
  WorkerId GetWorkerId() const { return worker_id; }
  const std::set<int>& GetUnitIndices() const { return unit_indices; }
  std::string GetUnitIndicesString() const;

  std::string ToString() const;
  bool IsValid() const;

 private:
  ModelId model_id = -1;
  WorkerId worker_id = -1;
  std::set<int> unit_indices;
};

// hash function to use SubgraphKey as a key
// https://stackoverflow.com/a/32685618
struct SubgraphHash {
  std::size_t operator()(const SubgraphKey& p) const;
};

enum JobStatus {
  kBandJobQueued,
  kBandJobSuccess,
  kBandJobSLOViolation,
  kBandJobInputCopyFailure,
  kBandJobOutputCopyFailure,
  kBandJobInvokeFailure
};

// Job struct is the scheduling and executing unit.
// The request can specify a model by indication the model id
struct Job {
  explicit Job() : model_id(-1) {}
  explicit Job(ModelId model_id) : model_id(model_id) {}
  explicit Job(ModelId model_id, int64_t slo)
      : model_id(model_id), slo_us(slo) {}

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
  int64_t slo_us = 0;

  // Target worker id (only for fixed worker request)
  WorkerId target_worker_id = -1;

  // Constant variables (Valid after invoke)
  // TODO: better job life-cycle to change these to `const`
  ModelId model_id;
  int input_handle = -1;
  int output_handle = -1;
  JobId job_id = -1;
  int sched_id = -1;
  std::string model_fname;
  bool require_callback = true;

  // Current status for execution (Valid after planning)
  JobStatus status = kBandJobQueued;
  SubgraphKey subgraph_key;
  int start_unit_idx = 0;
  std::vector<Job> following_jobs;
  // see Interpreter::MakeSubgraphsForFallbackOps for details on this field
  std::set<int> resolved_tensors;
  std::list<int> previous_subgraph_indices;
};

// hash function to use pair<int, set<int>> as map key in cache_
// https://stackoverflow.com/a/32685618
struct PairHash {
  std::size_t operator()(const std::pair<int, std::set<int>>& p) const;
};

}  // namespace Band

#endif  // BAND_COMMON_H_