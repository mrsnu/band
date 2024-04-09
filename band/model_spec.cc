#include "band/model_spec.h"

#include <algorithm>
#include <iterator>

#include "band/logger.h"

namespace band {
std::set<int> ModelSpec::GetPureInputTensors(
    const std::set<int>& op_indices) const {
  // {all input tensors in ops} - {all output tensors in ops}
  // 纯输入张量”指的是那些作为输入但不作为任何操作的输出的张量
  std::set<int> input_tensors;
  for (const auto& op_index : op_indices) {
    // 收集所有输入张量
    const std::set<int>& inputs = op_input_tensors[op_index];
    // 对于每一个操作索引op_index，函数查找对应的输入张量集合inputs
    input_tensors.insert(inputs.begin(), inputs.end());
    // 存储所有的输入张量
  }

  for (const auto& op_index : op_indices) {
    const std::set<int>& outputs = op_output_tensors[op_index];
    for (int output_index : outputs) {
      input_tensors.erase(output_index);
      // 去除输出张量
    }
  }

  return input_tensors;
}

std::set<int> ModelSpec::GetOutputTensors(
    const std::set<int>& op_indices) const {
  // {all output tensors in ops}
  std::set<int> output_tensors;
  for (const auto& op_index : op_indices) {
    const std::set<int>& outputs = op_output_tensors[op_index];
    output_tensors.insert(outputs.begin(), outputs.end());
  }

  return output_tensors;
}

/**
 * @brief Sets the unit subgraphs for the model specification.
 * 
 * This function sets the unit subgraphs for the model specification based on the provided operations.
 * 
 * @param ops A vector of sets of integers representing the operations in each unit subgraph.
 * @return absl::Status The status of the operation. Returns absl::OkStatus() if successful, or an absl::InternalError if the unit subgraph does not cover all operators.
 */
absl::Status band::ModelSpec::SetUnitSubgraphs(std::vector<std::set<int>> ops) {
  unit_subgraph_ops = ops;

  // Verify whether unit subgraph covers all ops
  // 验证提供的单元子图是否涵盖了模型中的所有操作，并且建立单元子图之间基于张量依赖的依赖关系。
  std::set<int> all_ops;
  for (const auto& unit_subgraph_ops_ : unit_subgraph_ops) {
    all_ops.insert(unit_subgraph_ops_.begin(), unit_subgraph_ops_.end());
  }

  if ((all_ops.size() != num_ops) || (*all_ops.rbegin() != num_ops - 1)) {
    return absl::InternalError(
        "Failed to set unit subgraphs. Unit subgraph does not covers "
        "all operators");
  }

  unit_subgraph_dependencies.resize(ops.size());
  // 计算单元子图之间的依赖关系
  for (int child = 0; child < unit_subgraph_ops.size(); child++) {
    for (int potential_parent = 0; potential_parent < child;
         potential_parent++) {
      // Mark as dependent based on a tensor dependency
      // 对于每个单元子图（称为child），检查它与所有索引小于它的单元子图（称为potential_parent）之间是否存在依赖关系。
      std::set<int> intersection;
      auto child_inputs = GetPureInputTensors(unit_subgraph_ops[child]);
      auto parent_outputs =
          GetOutputTensors(unit_subgraph_ops[potential_parent]);
      std::set_intersection(child_inputs.begin(), child_inputs.end(),
                            parent_outputs.begin(), parent_outputs.end(),
                            std::inserter(intersection, intersection.begin()));
      if (intersection.size()) {
        unit_subgraph_dependencies[child].set(potential_parent);
      }
      // 计算依赖关系的方法是找出child单元子图的纯输入张量与potential_parent单元子图的输出张量之间的交集。
      //  如果存在交集，表示child依赖于potential_parent，在依赖关系数据结构中做相应的标记。
    }
  }

  return absl::OkStatus();
}

size_t ModelSpec::GetNumUnitSubgraphs() const {
  return unit_subgraph_ops.size();
}

const std::set<int>& ModelSpec::GetUnitSubgraphOps(size_t index) const {
  return unit_subgraph_ops[index];
}

const BitMask& ModelSpec::GetUnitSubgraphDependency(size_t index) const {
  return unit_subgraph_dependencies[index];
}

BitMask ModelSpec::GetUnitSubgraphDependency(
    const BitMask& unit_subgraphs) const {
  BitMask external_dependencies;
  // get all dependencies to run given unit subgraphs
  for (size_t i = 0; i < GetNumUnitSubgraphs(); i++) {
    if (unit_subgraphs.test(i)) {
      external_dependencies |= GetUnitSubgraphDependency(i);
    }
  }
  // remove any internal dependencies
  external_dependencies &= ~unit_subgraphs;
  return external_dependencies;
}

}  // namespace band