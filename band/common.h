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
  // special case - entire model subgraph
  SubgraphKey(ModelId model_id, WorkerId worker_id);
  SubgraphKey(ModelId model_id, WorkerId worker_id, std::set<int> input_ops,
              std::set<int> output_ops);
  bool operator<(const SubgraphKey& key) const;
  bool operator==(const SubgraphKey& key) const;

  std::string GetInputOpsString() const;
  std::string GetOutputOpsString() const;

  ModelId GetModelId() const { return model_id; }
  WorkerId GetWorkerId() const { return worker_id; }
  const std::set<int>& GetInputOps() const { return input_ops; }
  const std::set<int>& GetOutputOps() const { return output_ops; }

  bool IsValid() const;

 private:
  ModelId model_id = -1;
  WorkerId worker_id = -1;
  std::set<int> input_ops;
  std::set<int> output_ops;

  // TODO: Where to move `unit_indices`?
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

  // Constant variables (Valid after invoke)
  // TODO: better job life-cycle to change these to `const`
  ModelId model_id;
  int input_handle = -1;
  int output_handle = -1;
  JobId job_id = -1;
  int sched_id = -1;
  std::string model_fname;

  // Current status for execution (Valid after planning)
  JobStatus status = kBandJobQueued;
  SubgraphKey subgraph_key;
  int start_unit_idx = 0;
  std::vector<Job> following_jobs;
  // see Interpreter::MakeSubgraphsForFallbackOps for details on this field
  std::set<int> resolved_tensors;
  std::list<int> previous_subgraph_indices;
};

// a convenient data structure for holding various model information
struct ModelSpec {
  int num_ops;
  std::set<int> input_tensors;
  // only includes "true" outputs
  std::set<int> output_tensors;
  // includes intermediate tensors that are consumed by
  // other nodes in the same model
  std::set<int> node_output_tensors;
  std::set<BandType> tensor_types;
  std::map<BandDeviceFlags, std::set<int>> unsupported_ops;
  int num_unit_subgraphs;
  // vector for memoization during scheduling.
  // Each element is a pair of subgraph indices list and shortest latency.
  std::vector<std::pair<std::vector<int>, int64_t>> latency_memo;
};

// hash function to use pair<int, set<int>> as map key in cache_
// https://stackoverflow.com/a/32685618
struct PairHash {
  std::size_t operator()(const std::pair<int, std::set<int>>& p) const;
};

}  // namespace Band

#endif  // BAND_COMMON_H_