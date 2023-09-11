// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/model_spec.h"

#include <algorithm>
#include <iterator>

#include "band/logger.h"

namespace band {
std::set<int> ModelSpec::GetPureInputTensors(
    const std::set<int>& op_indices) const {
  // {all input tensors in ops} - {all output tensors in ops}
  std::set<int> input_tensors;
  for (const auto& op_index : op_indices) {
    const std::set<int>& inputs = op_input_tensors[op_index];
    input_tensors.insert(inputs.begin(), inputs.end());
  }

  for (const auto& op_index : op_indices) {
    const std::set<int>& outputs = op_output_tensors[op_index];
    for (int output_index : outputs) {
      input_tensors.erase(output_index);
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

absl::Status band::ModelSpec::SetUnitSubgraphs(std::vector<std::set<int>> ops) {
  unit_subgraph_ops = ops;

  // Verify whether unit subgraph covers all ops
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

  for (int child = 0; child < unit_subgraph_ops.size(); child++) {
    for (int potential_parent = 0; potential_parent < child;
         potential_parent++) {
      // Mark as dependent based on a tensor dependency
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