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

// a convenient data structure for holding various model information
struct ModelSpec {
  // explicitly remove default ctor, to force initialization of required
  // params
  ModelSpec() : ModelSpec(0, 0, {}, {}, {}, {}, {}, {}, {}) {}
  ModelSpec(int num_ops, int num_tensors, std::vector<BandType> tensor_types,
            std::set<int> input_tensors, std::set<int> output_tensors,
            std::vector<std::set<int>> op_input_tensors,
            std::vector<std::set<int>> op_output_tensors,
            std::map<BandDeviceFlags, std::set<int>> unsupported_ops,
            std::set<BandDeviceFlags> unavailable_devices)
      : num_ops(num_ops),
        num_tensors(num_tensors),
        tensor_types(tensor_types),
        input_tensors(input_tensors),
        output_tensors(output_tensors),
        op_input_tensors(op_input_tensors),
        op_output_tensors(op_output_tensors),
        unsupported_ops(unsupported_ops),
        unavailable_devices(unavailable_devices) {}

  // Get `pure` input tensors to given subgraph
  // that requires external dependency from predecessors.
  std::set<int> GetPureInputTensors(const std::set<int>& op_indices) const;
  // Get all output tensors from all ops in a given subgraph,
  // We can't compute a `pure` output tensor since there is no information on
  // whether a particular op's output is pointing external op. (e.g.,
  // lite-model_efficientdet_lite0_int8_1.tflite`s 64'th node (MaxPool2D)
  // connected to multiple ops across multiple subgraphs in Pixel 4 -- output
  // tensor #396).
  std::set<int> GetOutputTensors(const std::set<int>& op_indices) const;

  /* from Interpreter::InvestigateModelSpec */
  const int num_ops;
  const int num_tensors;
  const std::vector<BandType> tensor_types;
  // indices to input / output tensors
  const std::set<int> input_tensors;
  const std::set<int> output_tensors;

  // includes intermediate tensors that are provided /consumed by
  // other ops in the same model
  // NOTE: remove the ones from model definition / weights
  // e.g., kTfLiteMmapRo in Tensorflow Lite
  const std::vector<std::set<int>> op_input_tensors;
  const std::vector<std::set<int>> op_output_tensors;
  const std::map<BandDeviceFlags, std::set<int>> unsupported_ops;
  const std::set<BandDeviceFlags> unavailable_devices;

  std::string path;

  /* from ModelAnalyzer */
  std::vector<std::set<int>> unit_subgraph_ops;
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