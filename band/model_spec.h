#ifndef BAND_MODEL_SPEC_H_
#define BAND_MODEL_SPEC_H_

#include <set>
#include <string>
#include <vector>

#include "band/common.h"

namespace Band {
// a convenient data structure for holding various model information
class ModelSpec {
 public:
  // explicitly remove default ctor, to force initialization of required
  // params
  ModelSpec() : ModelSpec(0, 0, {}, {}, {}, {}, {}, {}, {}) {}
  ModelSpec(int num_ops, int num_tensors, std::vector<DataType> tensor_types,
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
  BandStatus SetUnitSubgraphs(std::vector<std::set<int>> ops);

  size_t GetNumUnitSubgraphs() const;
  const std::set<int>& GetUnitSubgraphOps(size_t index) const;
  const BitMask& GetUnitSubgraphDependency(size_t index) const;
  BitMask GetUnitSubgraphDependency(const BitMask& unit_subgraphs) const;

  /* from Interpreter::InvestigateModelSpec */
  const int num_ops;
  const int num_tensors;
  const std::vector<DataType> tensor_types;
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

 private:
  std::vector<std::set<int>> unit_subgraph_ops;
  // `direct` dependency table between unit subgraphs. e.g., if unit subgraph 2
  // depends on 0 and 1, unit_subgraph_dependencies[2] =
  // ...0011
  std::vector<BitMask> unit_subgraph_dependencies;

  // vector for memoization during scheduling.
  // Each element is a pair of subgraph indices list and shortest latency.
  std::vector<std::pair<std::vector<int>, int64_t>> latency_memo;
};
}  // namespace Band

#endif