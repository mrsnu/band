#ifndef BAND_MODEL_ANALYZER_H_
#define BAND_MODEL_ANALYZER_H_

#include <map>
#include <memory>

#include "band/context.h"

namespace Band {
class Model;

struct SubgraphDef {
  WorkerId worker_id;
  std::set<int> op_indices;
  std::set<int> unit_subgraph_indices;
  std::string ToString() const;
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

  std::vector<std::set<int>> unit_subgraph_ops;
  /* from ModelAnalyzer */
  int num_unit_subgraphs;
  // vector for memoization during scheduling.
  // Each element is a pair of subgraph indices list and shortest latency.
  std::vector<std::pair<std::vector<int>, int64_t>> latency_memo;
};

std::string SetToString(const std::set<int>& set);
std::string SummarizeSubgraphs(const std::vector<SubgraphDef>& subgraph_defs);
std::string SummarizeFallbackPerWorkerSubgraphs(
    const std::vector<SubgraphDef>& unit_subgraph_defs,
    const std::vector<SubgraphDef>& subgraph_defs);

class ModelAnalyzer {
 public:
  ModelAnalyzer(const Context& context, bool need_subgraph,
                ModelConfig model_config, Model* model,
                BandBackendType backend_type);

  std::tuple<BandStatus, ModelSpec, std::vector<SubgraphDef>> CreateSubgraphs();

 private:
  // A model is partitioned into unit subgraphs.
  // We assign an index to each unit subgraph, and the unit subgraph indices are
  // topologically sorted. Note that there can be better way to assign unit
  // subgraph indices if there exists any unit subgraphs that can be executed in
  // parallel.
  BandStatus GetUnitSubgraphs(std::vector<SubgraphDef>& unit_subgraphs);
  // Generate subgraphs for fallback ops in provided model
  // This does not provides unit indices with a SubgraphDef
  std::vector<SubgraphDef> GetSubgraphsForFallbackOps(WorkerId worker_id);
  std::vector<SubgraphDef> MergeUnitSubgraphs(
      const std::vector<SubgraphDef>& unit_subgraphs);

  bool NeedFallbackSubgraph() const;
  bool IsWorkerValid(WorkerId worker_id) const;
  bool IsResolved(const std::set<int> resolved_tensors, int op_index) const;

  const Context& context_;
  const bool need_fallback_subgraph_;
  const ModelConfig model_config_;
  const BandBackendType backend_type_;
  std::shared_ptr<ModelSpec> model_spec_;
};
}  // namespace Band

#endif