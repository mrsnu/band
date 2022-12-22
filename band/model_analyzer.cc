#include "band/model_analyzer.h"

#include <memory>

#include "band/backend_factory.h"
#include "band/context.h"
#include "band/interface/interpreter.h"
#include "band/interface/model.h"
#include "band/model.h"
#include "band/worker.h"
#include "model_analyzer.h"

namespace Band {
ModelAnalyzer::ModelAnalyzer(const Context& context, ModelConfig model_config,
                             Model* model, BandBackendType backend_type)
    : context_(context),
      model_config_(model_config),
      backend_type_(backend_type) {
  std::unique_ptr<Interface::IInterpreter> interpreter(
      BackendFactory::CreateInterpreter(backend_type));
  model_spec_ = std::make_shared<ModelSpec>(
      interpreter->InvestigateModelSpec(model->GetBackendModel(backend_type)));
}

// TODO: do PrepareUnitSubgraphScheduling after this
BandStatus ModelAnalyzer::GetUnitSubgraphs(
    ModelId model_id,
    std::vector<std::pair<int, SubgraphDef>>& unit_subgraphs) {
  const int num_workers = context_.GetNumWorkers();

  if (model_config_.subgraph_preparation_type == kBandNoFallbackSubgraph) {
    std::set<int> entire_ops;
    for (int i = 0; i < model_spec_->num_ops; i++) {
      entire_ops.insert(i);
    }
    for (WorkerId id = 0; id < num_workers; id++) {
      unit_subgraphs.push_back({0, {id, entire_ops}});
    }
    return kBandOk;
  }

  const int num_ops = model_spec_->num_ops;

  using BitMask = uint32_t;
  if (num_workers > 8 * sizeof(BitMask)) {
    BAND_REPORT_ERROR(context_.GetErrorReporter(),
                      "Number of workers is larger than BitMask %d",
                      num_workers);
    return kBandError;
  }

  std::map<WorkerId, std::set<int>> op_sets_to_ignore;
  // register subgraphs for all workers
  for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
    std::vector<SubgraphDef> worker_op_sets =
        MakeSubgraphsForFallbackOps(model_id, worker_id);
    for (auto worker_and_ops : worker_op_sets) {
      if (context_.GetWorker(worker_id)->GetDeviceFlag() == kBandCPU) {
        continue;
      }
      if (worker_and_ops.op_indices.size() <
          model_config_.minimum_subgraph_size) {
        for (auto op : worker_and_ops.op_indices) {
          op_sets_to_ignore[worker_id].insert(op);
        }
      }
    }
  }

  // build op_support_table
  std::vector<BitMask> op_support_table(num_ops, 0U);
  std::map<WorkerId, std::set<int>> unsupported_ops;
  // TODO(BAND-62): assume that band device type targets a single processor.
  for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
    unsupported_ops[worker_id] = model_spec_->unsupported_ops.at(
        context_.GetWorker(worker_id)->GetDeviceFlag());
  }

  for (int op_index = 0; op_index < num_ops; op_index++) {
    for (WorkerId worker_id = 0; worker_id < num_workers; ++worker_id) {
      BandDeviceFlags device_flag =
          context_.GetWorker(worker_id)->GetDeviceFlag();
      if (device_flag == kBandCPU) {
        op_support_table[op_index] |= 1 << worker_id;
        continue;
      }

      if (unsupported_ops.find(device_flag) == unsupported_ops.end() ||
          unsupported_ops.at(device_flag).find(op_index) ==
              unsupported_ops.at(device_flag).end()) {
        if (op_sets_to_ignore[device_flag].find(op_index) ==
            op_sets_to_ignore[device_flag].end()) {
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

  int subgraph_local_idx = 0;
  while (true) {
    std::set<int> unit_subgraph_ops;
    BitMask support_devices = 0;

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
        if (support_devices != 0 &&
            support_devices != op_support_table[op_index]) {
          continue;
        }
        // Set support devices using first op
        if (support_devices == 0) {
          support_devices = op_support_table[op_index];
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
      if (support_devices & (1 << worker_id)) {
        unit_subgraphs.push_back(
            {subgraph_local_idx, {worker_id, unit_subgraph_ops}});
      }
    }
    subgraph_local_idx++;
  }

  if (!remaining_ops.empty()) {
    BAND_REPORT_ERROR(context_.GetErrorReporter(), "Not empty remaining ops");
    return kBandError;
  }

  return kBandOk;
}

std::vector<SubgraphDef> Band::ModelAnalyzer::MakeSubgraphsForFallbackOps(
    ModelId model_id, WorkerId worker_id) {
  const Worker* worker = context_.GetWorker(worker_id);
  if (!worker) {
    BAND_REPORT_ERROR(context_.GetErrorReporter(), "Invalied worker_id %d",
                      worker_id);
    return {};
  }

  if (model_config_.subgraph_preparation_type == kBandNoFallbackSubgraph) {
    std::set<int> entire_ops;
    for (int i = 0; i < model_spec_->num_ops; i++) {
      entire_ops.insert(i);
    }
    return {{worker_id, entire_ops}};
  }

  std::vector<SubgraphDef> subgraph_indices;
  const int num_ops = model_spec_->num_ops;
  const BandDeviceFlags device_flag =
      context_.GetWorker(worker_id)->GetDeviceFlag();
  const std::set<int>& unsupported_ops =
      model_spec_->unsupported_ops.at(device_flag);

  std::set<int> resolved_tensors;
  std::set<int> remaining_ops;
  // The basic idea is to partition this model into several disjoint subgraphs.
  // Each subgraph is not necessarily a connected graph, and no two graphs
  // have any common ops. A subgraph is either a fallback subgraph or a
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

  // To make this work, we first need to keep track of the "front line" of ops.
  // This front line, together with the fallback/non-fb status of the op,
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
  while (remaining_ops.size() > 0) {
    std::set<int> operator_set;
    bool found = true;
    // Switch between device and fallback
    BandDeviceFlags current_device = is_fallback ? kBandCPU : device_flag;

    // Get all op that has resolvable dependency to specific device
    while (found) {
      found = false;
      for (auto current_op = remaining_ops.begin();
           current_op != remaining_ops.end();) {
        int current_index = *current_op;
        bool is_op_unsupported =
            unsupported_ops.find(current_index) != unsupported_ops.end();
        if (!is_fallback == is_op_unsupported) {
          // either 1) this is a fallback op but we're making a non-fb subgraph,
          // or 2) this is a non-fb op but we're making a fb subgraph,
          // so we skip it
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
      subgraph_indices.push_back({current_device, operator_set});
    }

    is_fallback = !is_fallback;
  }

  return subgraph_indices;
}

const ModelSpec& ModelAnalyzer::GetModelSpec() const { return *model_spec_; }

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
}  // namespace Band