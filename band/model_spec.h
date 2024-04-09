#ifndef BAND_MODEL_SPEC_H_
#define BAND_MODEL_SPEC_H_

#include <set>
#include <string>
#include <vector>

#include "band/common.h"

#include "absl/status/status.h"

namespace band {
// a convenient data structure for holding various model information
// 这是一个便于存储各种模型信息的实用数据结构。
class ModelSpec {
 public:
  // explicitly remove default ctor, to force initialization of required
  // params
  // 为了确保必须初始化所有必需的参数，我们故意不提供默认的构造函数。
  ModelSpec() : ModelSpec(0, 0, {}, {}, {}, {}, {}, {}, {}) {}
  ModelSpec(int num_ops, int num_tensors, std::vector<DataType> tensor_types,
            std::set<int> input_tensors, std::set<int> output_tensors,
            std::vector<std::set<int>> op_input_tensors,
            std::vector<std::set<int>> op_output_tensors,
            std::map<DeviceFlag, std::set<int>> unsupported_ops,
            std::set<DeviceFlag> unavailable_devices)
      : num_ops(num_ops),
      // 模型中的操作数
        num_tensors(num_tensors),
        // 模型中的张量数
        tensor_types(tensor_types),
        // 张量类型
        input_tensors(input_tensors),
        output_tensors(output_tensors),
        op_input_tensors(op_input_tensors),
        op_output_tensors(op_output_tensors),
        unsupported_ops(unsupported_ops),
        unavailable_devices(unavailable_devices) {}

  // Get `pure` input tensors to given subgraph
  // that requires external dependency from predecessors.
  // 该结构可以获取给定子图的“纯净”输入张量，这些输入张量需要依赖于前面的操作。
  std::set<int> GetPureInputTensors(const std::set<int>& op_indices) const;
  // Get all output tensors from all ops in a given subgraph,
  // We can't compute a `pure` output tensor since there is no information on
  // whether a particular op's output is pointing external op. (e.g.,
  // lite-model_efficientdet_lite0_int8_1.tflite`s 64'th node (MaxPool2D)
  // connected to multiple ops across multiple subgraphs in Pixel 4 -- output
  // tensor #396).
  // 它还能从给定子图的所有操作中获取所有输出张量。但我们不能确定输出张量是否“纯净”，因为缺乏信息判断某个操作的输出是否指向了外部操作（
  // 比如，在 Pixel 4 设备上，lite-model_efficientdet_lite0_int8_1.tflite 的第64个节点（MaxPool2D）可能会与多个子图中的多个操作相连——例如输出张量 #396）。
  std::set<int> GetOutputTensors(const std::set<int>& op_indices) const;
  absl::Status SetUnitSubgraphs(std::vector<std::set<int>> ops);

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
  // 此外，还包括在同一模型中由其他操作使用或产生的中间张量。
  // 注意：从模型定义或权重中移除这些张量。例如，Tensorflow Lite 中的 kTfLiteMmapRo。
  const std::vector<std::set<int>> op_input_tensors;
  const std::vector<std::set<int>> op_output_tensors;
  const std::map<DeviceFlag, std::set<int>> unsupported_ops;
  const std::set<DeviceFlag> unavailable_devices;

  std::string path;

 private:
  std::vector<std::set<int>> unit_subgraph_ops;
  // `direct` dependency table between unit subgraphs. e.g., if unit subgraph 2
  // depends on 0 and 1, unit_subgraph_dependencies[2] =
  // ...0011
  // 这是一个描述单元子图间直接依赖关系的表。举个例子，如果单元子图 2 依赖于单元子图 0 和 1，
  // 则 unit_subgraph_dependencies[2] 的值为 ...0011。
  std::vector<BitMask> unit_subgraph_dependencies;

  // vector for memoization during scheduling.
  // Each element is a pair of subgraph indices list and shortest latency.
  // 在调度过程中用于存储备忘录的向量。每个元素都是一个子图索引列表和最短延迟的对。
  std::vector<std::pair<std::vector<int>, int64_t>> latency_memo;
};
}  // namespace band

#endif