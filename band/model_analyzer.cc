#include "band/model_analyzer.h"

#include <algorithm>
#include <iterator>
#include <memory>

#include "absl/strings/str_format.h"
#include "band/backend_factory.h"
#include "band/context.h"
#include "band/interface/model.h"
#include "band/interface/model_executor.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/worker.h"

namespace band {
std::string SetToString(const std::set<int>& set) {
  auto range_to_string = [](int lhs, int rhs) {
    if (lhs == rhs) {
      return std::to_string(lhs);
    } else {
      return std::to_string(lhs) + "-" + std::to_string(rhs);
    }
  };

  std::string result = "{";
  if (set.size() > 0) {
    int current_start = std::numeric_limits<int>::min();
    int prev = current_start;
    for (auto v : set) {
      // not continuous
      if (v > prev + 1) {
        if (current_start >= 0) {
          result += range_to_string(current_start, prev) + ",";
        }
        current_start = v;
      }
      prev = v;
    }
    result += range_to_string(current_start, *set.rbegin());
  }
  return result + "}";
}

std::string SubgraphDef::ToString() const {
  return "Index " + SetToString(unit_subgraph_indices) + " Ops " +
         SetToString(op_indices);
}

std::string SummarizeSubgraphs(const std::vector<SubgraphDef>& subgraph_defs) {
  std::string summary = "\n";
  std::vector<SubgraphDef> unit_subgraphs;
  std::vector<SubgraphDef> merged_subgraphs;
  std::set<int> unique_unit_subgraph_indices;
  int num_workers = 0;
  for (const auto& subgraph_def : subgraph_defs) {
    if (subgraph_def.unit_subgraph_indices.size() == 1) {
      unit_subgraphs.push_back(subgraph_def);
      unique_unit_subgraph_indices.insert(
          *subgraph_def.unit_subgraph_indices.begin());
    } else {
      merged_subgraphs.push_back(subgraph_def);
    }
    num_workers = std::max(num_workers, subgraph_def.worker_id + 1);
  }

  if (unit_subgraphs.size()) {
    summary += "UnitSubgraph Definitions\n";

    std::map<WorkerId, std::vector<bool>> unit_subgraph_availabilities;
    for (WorkerId i = 0; i < num_workers; i++) {
      unit_subgraph_availabilities[i] =
          std::vector<bool>(unique_unit_subgraph_indices.size(), false);
    }

    for (const auto& unit_subgraph : unit_subgraphs) {
      unit_subgraph_availabilities[unit_subgraph.worker_id]
                                  [*unit_subgraph.unit_subgraph_indices
                                        .begin()] = true;
      if (unit_subgraph.worker_id == 0) {
        summary += "\t" + unit_subgraph.ToString() + "\n";
      }
    }

    summary += "UnitSubgraph Availabilities\n";

    for (const auto& unit_subgraph_availability :
         unit_subgraph_availabilities) {
      summary += "\t Worker " +
                 std::to_string(unit_subgraph_availability.first) + "\t";
      for (const auto& availability : unit_subgraph_availability.second) {
        summary += (availability ? "O\t" : "X\t");
      }
      summary += "\n";
    }
  }

  if (merged_subgraphs.size()) {
    summary += "MergedSubgraphs\n";

    for (WorkerId target_worker_id = 0; target_worker_id < num_workers;
         target_worker_id++) {
      for (const auto& merged_subgraph : merged_subgraphs) {
        if (merged_subgraph.worker_id == target_worker_id) {
          summary += "\t Worker " + std::to_string(target_worker_id) + "\t";
          for (const auto& unit_index : unique_unit_subgraph_indices) {
            summary +=
                (merged_subgraph.unit_subgraph_indices.find(unit_index) !=
                         merged_subgraph.unit_subgraph_indices.end()
                     ? "-\t"
                     : " \t");
          }
          summary += "\n";
        }
      }
    }
  }

  return summary;
}

std::string SummarizeFallbackPerWorkerSubgraphs(
    const std::vector<SubgraphDef>& unit_subgraph_defs,
    const std::vector<SubgraphDef>& subgraph_defs) {
  std::string summary = SummarizeSubgraphs(unit_subgraph_defs);

  std::set<int> unique_unit_subgraph_indices;
  int num_workers = 0;
  for (const auto& subgraph_def : unit_subgraph_defs) {
    if (subgraph_def.unit_subgraph_indices.size() == 1) {
      unique_unit_subgraph_indices.insert(
          *subgraph_def.unit_subgraph_indices.begin());
    }
    num_workers = std::max(num_workers, subgraph_def.worker_id + 1);
  }

  summary += "FallbackPerWorkerSubgraphs\n";

  for (WorkerId target_worker_id = 0; target_worker_id < num_workers;
       target_worker_id++) {
    for (const auto& merged_subgraph : subgraph_defs) {
      if (merged_subgraph.worker_id == target_worker_id) {
        summary += "\t Worker " + std::to_string(target_worker_id) + "\t";
        for (const auto& unit_index : unique_unit_subgraph_indices) {
          summary += (merged_subgraph.unit_subgraph_indices.find(unit_index) !=
                              merged_subgraph.unit_subgraph_indices.end()
                          ? "-\t"
                          : " \t");
        }
        summary += "\n";
      }
    }
  }

  return summary;
}

ModelAnalyzer::ModelAnalyzer(const Context& context,
                             bool need_fallback_subgraph,
                             SubgraphConfig subgraph_config, Model* model,
                             BackendType backend_type)
    : context_(context),
      need_fallback_subgraph_(need_fallback_subgraph),
      subgraph_config_(subgraph_config),
      backend_type_(backend_type) {
  std::unique_ptr<interface::IModelExecutor> interpreter(
      BackendFactory::CreateModelExecutor(backend_type, model->GetId(), 0,
                                          DeviceFlags::CPU));
  // TODO(widiba03304): Report error when it fails.
  model_spec_ = std::make_shared<ModelSpec>(
      interpreter->InvestigateModelSpec(model->GetBackendModel(backend_type))
          .value());

  for (auto device_unsupported_ops : model_spec_->unsupported_ops) {
    BAND_LOG_INFO("Unsupported ops %s (%s)",
                  SetToString(device_unsupported_ops.second).c_str(),
                  GetName(device_unsupported_ops.first).c_str());
  }

  for (auto device : model_spec_->unavailable_devices) {
    BAND_LOG_INFO("Unsupported devices %s", GetName(device).c_str());
  }
}

absl::StatusOr<std::pair<ModelSpec, std::vector<SubgraphDef>>>
ModelAnalyzer::CreateSubgraphs() {
  std::vector<SubgraphDef> subgraph_defs;
  std::vector<SubgraphDef> unit_subgraph_defs;

  // TODO(widiba03304): Add error propagation logic.
  auto status = GetUnitSubgraphs(unit_subgraph_defs);
  if (!status.ok()) {
    return status;
  }

  switch (subgraph_config_.subgraph_preparation_type) {
    case SubgraphPreparationType::FallbackPerWorker: {
      for (WorkerId worker_id = 0; worker_id < context_.GetNumWorkers();
           worker_id++) {
        std::vector<SubgraphDef> worker_subgraphs =
            GetSubgraphsForFallbackOps(worker_id);

        for (SubgraphDef& worker_subgraph : worker_subgraphs) {
          // set unit subgraph indices
          for (int unit_subgraph_id = 0;
               unit_subgraph_id < unit_subgraph_defs.size();
               unit_subgraph_id++) {
            // add all unit subgraphs that are part of the worker subgraph
            if (std::includes(
                    worker_subgraph.op_indices.begin(),
                    worker_subgraph.op_indices.end(),
                    unit_subgraph_defs[unit_subgraph_id].op_indices.begin(),
                    unit_subgraph_defs[unit_subgraph_id].op_indices.end())) {
              worker_subgraph.unit_subgraph_indices.insert(
                  unit_subgraph_defs[unit_subgraph_id]
                      .unit_subgraph_indices.begin(),
                  unit_subgraph_defs[unit_subgraph_id]
                      .unit_subgraph_indices.end());
            }
          }
        }

        subgraph_defs.insert(subgraph_defs.end(), worker_subgraphs.begin(),
                             worker_subgraphs.end());
      }
    } break;
    case SubgraphPreparationType::NoFallbackSubgraph:
    case SubgraphPreparationType::UnitSubgraph: {
      subgraph_defs = unit_subgraph_defs;
    } break;
    case SubgraphPreparationType::MergeUnitSubgraph: {
      // Add merged atomic subgraphs
      // Note that each merged subgraph consists of unit subgraphs with
      // continuous unit subgraph indices.
      // If we find any of the case that does not satisfy the condition,
      // we should re-implement the merging logic.
      subgraph_defs = MergeUnitSubgraphs(unit_subgraph_defs);
    } break;
  }

  // Verify subgraphs
  {
    // unit subgraph indices in merged subgraph are continous
    for (const auto& subgraph_def : subgraph_defs) {
      const int begin = *subgraph_def.unit_subgraph_indices.begin();
      const int end = *subgraph_def.unit_subgraph_indices.rbegin();
      if (end - begin != subgraph_def.unit_subgraph_indices.size() - 1) {
        return absl::InternalError(absl::StrFormat(
            "Failed to create subgraph. Unit subgraph indices in "
            "subgraph %s are not continous for model %s and mode %s",
            subgraph_def.ToString().c_str(), model_spec_->path.c_str(),
            GetName(subgraph_config_.subgraph_preparation_type)));
      }
    }
  }

  const std::string subgraph_summary =
      subgraph_config_.subgraph_preparation_type !=
              SubgraphPreparationType::FallbackPerWorker
          ? SummarizeSubgraphs(subgraph_defs)
          : SummarizeFallbackPerWorkerSubgraphs(unit_subgraph_defs,
                                                subgraph_defs);

  BAND_LOG_INFO("Create %d subgraphs for model %s with mode %s %s",
                subgraph_defs.size(), model_spec_->path.c_str(),
                GetName(subgraph_config_.subgraph_preparation_type).c_str(),
                subgraph_summary.c_str());

  return std::make_pair(*model_spec_, subgraph_defs);
}

absl::Status ModelAnalyzer::GetUnitSubgraphs(
    std::vector<SubgraphDef>& unit_subgraphs) {
  const int num_workers = context_.GetNumWorkers();
  unit_subgraphs.clear();

  if (!NeedFallbackSubgraph()) {
    std::set<int> entire_ops;
    for (int i = 0; i < model_spec_->num_ops; i++) {
      entire_ops.insert(i);
    }

    for (WorkerId worker_id = 0; worker_id < num_workers; worker_id++) {
      if (IsWorkerValid(worker_id)) {
        unit_subgraphs.push_back({worker_id, entire_ops, {0}});
      }
    }
  } else {
    const int num_ops = model_spec_->num_ops;
    if (num_workers > 8 * sizeof(BitMask)) {
      return absl::InternalError(absl::StrFormat(
          "Number of workers is larger than BitMask %d", num_workers));
    }

    std::map<WorkerId, std::set<int>> op_sets_to_ignore;
    // register subgraphs for all workers
    for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
      std::vector<SubgraphDef> worker_op_sets =
          GetSubgraphsForFallbackOps(worker_id);
      for (auto worker_and_ops : worker_op_sets) {
        if (context_.GetWorker(worker_id)->GetDeviceFlag() ==
            DeviceFlags::CPU) {
          continue;
        }
        if (worker_and_ops.op_indices.size() <
            subgraph_config_.minimum_subgraph_size) {
          for (auto op : worker_and_ops.op_indices) {
            op_sets_to_ignore[worker_id].insert(op);
          }
        }
      }
    }

    // build op_support_table
    std::vector<BitMask> op_support_table(num_ops, 0U);
    std::map<WorkerId, std::set<int>> unsupported_ops;
    int unit_subgraph_index = 0;
    // TODO(BAND-62): assume that band device type targets a single processor.
    for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
      if (IsWorkerValid(worker_id)) {
        unsupported_ops[worker_id] = model_spec_->unsupported_ops.at(
            context_.GetWorker(worker_id)->GetDeviceFlag());
      }
    }

    for (int op_index = 0; op_index < num_ops; op_index++) {
      for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
        DeviceFlags device_flag =
            context_.GetWorker(worker_id)->GetDeviceFlag();
        if (device_flag == DeviceFlags::CPU) {
          op_support_table[op_index] |= 1 << worker_id;
          continue;
        }

        WorkerId tmp_worker_id = static_cast<int>(device_flag);
        if (unsupported_ops.find(tmp_worker_id) == unsupported_ops.end() ||
            unsupported_ops.at(tmp_worker_id).find(op_index) ==
                unsupported_ops.at(tmp_worker_id).end()) {
          if (op_sets_to_ignore[tmp_worker_id].find(op_index) ==
              op_sets_to_ignore[tmp_worker_id].end()) {
            op_support_table[op_index] |= 1 << worker_id;
          }
        }
      }
    }

    // Add unit Subgraphs.
    // Op indices in single unit subgraph have same support devices.
    std::set<int> resolved_tensors;
    std::set<int> remaining_ops;

    for (int input_index : model_spec_->input_tensors) {
      resolved_tensors.insert(input_index);
    }

    for (int i = 0; i < num_ops; i++) {
      remaining_ops.insert(i);
    }

    while (true) {
      std::set<int> unit_subgraph_ops;
      BitMask support_workers = 0;

      // Find single unit subgraph ops
      while (true) {
        // Find addable ops
        // 1. resolved
        // 2. same support devices
        std::vector<int> to_add;
        for (int op_index : remaining_ops) {
          // Check the op is resolved
          if (!IsResolved(resolved_tensors, op_index)) {
            continue;
          }
          // Check the op have same support devices
          if (support_workers != 0 &&
              support_workers != op_support_table[op_index]) {
            continue;
          }
          // Set support devices using first op
          if (support_workers == 0) {
            support_workers = op_support_table[op_index];
          }
          to_add.push_back(op_index);
        }
        // If there is no more ops to add, stop
        if (to_add.empty()) break;

        // Add ops which are resolved and have same support devices
        unit_subgraph_ops.insert(to_add.begin(), to_add.end());

        // Delete resolved ops and add resolved tensors
        for (int op_index : to_add) {
          remaining_ops.erase(remaining_ops.find(op_index));
          const std::set<int>& op_outputs =
              model_spec_->op_output_tensors[op_index];
          for (int op_output_tensor : op_outputs) {
            resolved_tensors.insert(op_output_tensor);
          }
        }
      }
      if (unit_subgraph_ops.empty()) break;
      for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
        if (!IsWorkerValid(worker_id)) {
          continue;
        }
        if (support_workers.test(worker_id)) {
          unit_subgraphs.push_back(
              {worker_id, unit_subgraph_ops, {unit_subgraph_index}});
        }
      }
      unit_subgraph_index++;
    }

    if (!remaining_ops.empty()) {
      return absl::InternalError("Not empty remaining ops");
    }
  }

  std::set<int> unique_unit_subgraph_indices;

  for (const auto& subgraph_def : unit_subgraphs) {
    unique_unit_subgraph_indices.insert(
        *subgraph_def.unit_subgraph_indices.begin());
  }

  std::vector<std::set<int>> unit_subgraph_defs;

  unit_subgraph_defs.resize(unique_unit_subgraph_indices.size());
  for (const auto& unit_subgraph_def : unit_subgraphs) {
    unit_subgraph_defs[*unit_subgraph_def.unit_subgraph_indices.begin()] =
        unit_subgraph_def.op_indices;
  }

  // TODO(widiba03304): Add error propagation logic.
  auto status = model_spec_->SetUnitSubgraphs(unit_subgraph_defs);
  if (!status.ok()) {
    return status;
  }

  for (const auto& lhs : unit_subgraphs) {
    for (const auto& rhs : unit_subgraphs) {
      if (&lhs == &rhs) {
        continue;
      }

      if (*lhs.unit_subgraph_indices.begin() ==
          *rhs.unit_subgraph_indices.begin()) {
        if (lhs.op_indices != rhs.op_indices) {
          return absl::InternalError(absl::StrFormat(
              "Failed to create unit subgraph. Unit subgraph with same idx %d "
              "has different operators",
              *lhs.unit_subgraph_indices.begin()));
        }
      } else {
        std::set<int> intersection;
        std::set_intersection(
            lhs.op_indices.begin(), lhs.op_indices.end(),
            rhs.op_indices.begin(), rhs.op_indices.end(),
            std::inserter(intersection, intersection.begin()));
        if (intersection.size()) {
          return absl::InternalError(absl::StrFormat(
              "Failed to create unit subgraph. Unit subgraph with "
              "different idx %d, %d "
              "has common operators %s",
              *lhs.unit_subgraph_indices.begin(),
              *rhs.unit_subgraph_indices.begin(),
              SetToString(intersection).c_str()));
        }
      }
    }
  }

  BAND_LOG_INFO("Create %d unit subgraphs, planner requires subgraph %d",
                unique_unit_subgraph_indices.size(), NeedFallbackSubgraph());

  return absl::OkStatus();
}

std::vector<SubgraphDef> band::ModelAnalyzer::GetSubgraphsForFallbackOps(
    WorkerId worker_id) {
  const Worker* worker = context_.GetWorker(worker_id);
  if (!worker) {
    BAND_REPORT_ERROR(context_.GetErrorReporter(), "Invalied worker_id %d",
                      worker_id);
    return {};
  }

  if (!IsWorkerValid(worker_id)) {
    return {};
  }

  if (!NeedFallbackSubgraph()) {
    std::set<int> entire_ops;
    for (int i = 0; i < model_spec_->num_ops; i++) {
      entire_ops.insert(i);
    }
    return {{worker_id, entire_ops, {0}}};
  }

  std::vector<SubgraphDef> subgraph_defs;
  const int num_ops = model_spec_->num_ops;
  const DeviceFlags device_flag =
      context_.GetWorker(worker_id)->GetDeviceFlag();
  const std::set<int>& unsupported_ops =
      model_spec_->unsupported_ops.at(device_flag);

  std::set<int> cpu_worker_ids;
  for (WorkerId worker_id = 0; worker_id < context_.GetNumWorkers();
       worker_id++) {
    if (context_.GetWorker(worker_id)->GetDeviceFlag() == DeviceFlags::CPU) {
      cpu_worker_ids.insert(worker_id);
    }
  }

  std::set<int> resolved_tensors;
  std::set<int> remaining_ops;
  // The basic idea is to partition this model into several disjoint
  // subgraphs. Each subgraph is not necessarily a connected graph, and no two
  // graphs have any common ops. A subgraph is either a fallback subgraph or a
  // non-fallback one, but (obviously) never both.
  //
  //   Subgraph1  Sbg2     Sbg3
  // |--Non-fb--|--fb--|--Non-fb-|
  //
  //       Op2 --- Op3 -- Op4
  //     /                   \
  // Op1 - Op5 --- Op6 -- Op7 - Op8
  //
  // We start from the foremost op(s) and gradually "expand" our territory of
  // ops until we have the largest subgraph possible, without going over the
  // boundary of fallback/non-fallback. After that, we remove the ops of that
  // largest subgraph and start over with the remaining ops. This process is
  // repeated until all ops have been removed.

  // To make this work, we first need to keep track of the "front line" of
  // ops. This front line, together with the fallback/non-fb status of the op,
  // is used to determine whether or not we include an op in the current
  // subgraph.
  // The front line is denoted with the set of "resolved" tensors -- a tensor
  // is considered resolved if that tensor can be computed using external
  // inputs + previously resolved tensors. In case all input tensors of an
  // op are resolved ones, that op is regarded to be at the front line of ops
  // and thus can be put into the current subgraph (+ the fb/non-fb status
  // must match too).
  for (int input_index : model_spec_->input_tensors) {
    resolved_tensors.insert(input_index);
  }

  for (int i = 0; i < num_ops; i++) {
    remaining_ops.insert(i);
  }

  bool is_fallback = false;
  int unit_subgraph_idx = 0;
  while (remaining_ops.size() > 0) {
    std::set<int> operator_set;
    bool found = true;
    // Switch between device and fallback
    DeviceFlags current_device = is_fallback ? DeviceFlags::CPU : device_flag;

    // Get all op that has resolvable dependency to specific device
    while (found) {
      found = false;
      for (auto current_op = remaining_ops.begin();
           current_op != remaining_ops.end();) {
        int current_index = *current_op;
        bool is_op_unsupported =
            unsupported_ops.find(current_index) != unsupported_ops.end();
        if (!is_fallback == is_op_unsupported) {
          // either 1) this is a fallback op but we're making a non-fb
          // subgraph, or 2) this is a non-fb op but we're making a fb
          // subgraph, so we skip it
          current_op++;
          continue;
        }

        // Dependency check
        if (!IsResolved(resolved_tensors, current_index)) {
          current_op++;
          continue;
        }

        found = true;
        operator_set.insert(current_index);

        const std::set<int>& op_outputs =
            model_spec_->op_output_tensors[current_index];

        // Update dependency to include output tensors of this new op.
        // This has the effect of expanding the "front line" of ops.
        for (int op_output_tensor : op_outputs) {
          resolved_tensors.insert(op_output_tensor);
        }

        current_op = remaining_ops.erase(current_op);
      }
    }

    if (operator_set.size()) {
      if (current_device == DeviceFlags::CPU &&
          device_flag != DeviceFlags::CPU) {
        for (auto cpu_worker_id : cpu_worker_ids) {
          subgraph_defs.push_back({cpu_worker_id, operator_set, {}});
        }
      } else {
        subgraph_defs.push_back({worker_id, operator_set, {}});
      }
    }

    unit_subgraph_idx++;
    is_fallback = !is_fallback;
  }

  return subgraph_defs;
}

std::vector<SubgraphDef> ModelAnalyzer::MergeUnitSubgraphs(
    const std::vector<SubgraphDef>& unit_subgraphs) {
  std::vector<SubgraphDef> result_subgraphs = unit_subgraphs;

  // Check given worker - op_indices pair is already created or not
  auto is_already_created = [&result_subgraphs](WorkerId worker_id,
                                                std::set<int> op_indices) {
    for (const auto& subgraph : result_subgraphs) {
      if (subgraph.worker_id == worker_id &&
          subgraph.op_indices == op_indices) {
        return true;
      }
    }
    return false;
  };

  int num_subgraphs_before_merge = unit_subgraphs.size();
  bool added = true;
  while (added) {
    added = false;
    std::vector<SubgraphDef> subgraphs_to_add;
    for (const auto& prev_unit_subgraph : result_subgraphs) {
      const std::set<int> prev_outputs =
          model_spec_->GetOutputTensors(prev_unit_subgraph.op_indices);
      for (const auto& next_unit_subgraph : result_subgraphs) {
        // Prepare merged worker_id - op_indices
        const WorkerId worker_id = prev_unit_subgraph.worker_id;
        const std::set<int> next_inputs =
            model_spec_->GetPureInputTensors(next_unit_subgraph.op_indices);
        // Skip same subgraph or different device
        if ((&prev_unit_subgraph == &next_unit_subgraph) ||
            (prev_unit_subgraph.worker_id != next_unit_subgraph.worker_id)) {
          continue;
        }
        // Check whether prev subgraph fully resolves the next
        if (!std::includes(prev_outputs.begin(), prev_outputs.end(),
                           next_inputs.begin(), next_inputs.end())) {
          continue;
        }

        std::set<int> op_indices;
        const std::set<int>& prev_op_indices = prev_unit_subgraph.op_indices;
        const std::set<int>& next_op_indices = next_unit_subgraph.op_indices;
        std::set_union(prev_op_indices.begin(), prev_op_indices.end(),
                       next_op_indices.begin(), next_op_indices.end(),
                       std::inserter(op_indices, op_indices.end()));

        std::set<int> unit_subgraph_indices;
        std::set_union(
            prev_unit_subgraph.unit_subgraph_indices.begin(),
            prev_unit_subgraph.unit_subgraph_indices.end(),
            next_unit_subgraph.unit_subgraph_indices.begin(),
            next_unit_subgraph.unit_subgraph_indices.end(),
            std::inserter(unit_subgraph_indices, unit_subgraph_indices.end()));
        // Add if not already created
        if (!is_already_created(worker_id, op_indices)) {
          subgraphs_to_add.push_back(
              {worker_id, op_indices, unit_subgraph_indices});
        }
      }
    }

    for (auto& subgraph : subgraphs_to_add) {
      if (is_already_created(subgraph.worker_id, subgraph.op_indices)) continue;
      added = true;
      result_subgraphs.push_back(subgraph);
    }
  }

  BAND_LOG_INFO("Create %d merged subgraphs",
                result_subgraphs.size() - num_subgraphs_before_merge);

  return result_subgraphs;
}

bool ModelAnalyzer::NeedFallbackSubgraph() const {
  return need_fallback_subgraph_ &&
         (subgraph_config_.subgraph_preparation_type !=
          SubgraphPreparationType::NoFallbackSubgraph);
}

bool ModelAnalyzer::IsWorkerValid(WorkerId worker_id) const {
  return model_spec_->unavailable_devices.find(
             context_.GetWorker(worker_id)->GetDeviceFlag()) ==
         model_spec_->unavailable_devices.end();
}

bool ModelAnalyzer::IsResolved(const std::set<int> resolved_tensors,
                               int op_index) const {
  const std::set<int>& op_inputs = model_spec_->op_input_tensors[op_index];
  for (int op_input_tensor : op_inputs) {
    if (resolved_tensors.find(op_input_tensor) == resolved_tensors.end()) {
      return false;
    }
  }
  return true;
}
}  // namespace band