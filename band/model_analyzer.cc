#include "band/model_analyzer.h"

#include <memory>

#include "band/backend_factory.h"
#include "band/context.h"
#include "band/interface/interpreter.h"
#include "band/interface/model.h"
#include "band/model.h"

namespace Band {
ModelAnalyzer::ModelAnalyzer(const Context& context, Model* model,
                             BandBackendType backend_type)
    : context_(context), target_backend_type_(backend_type) {
  std::unique_ptr<Interface::IInterpreter> interpreter(
      BackendFactory::CreateInterpreter(backend_type));
  model_spec_ =
      interpreter->InvestigateModelSpec(model->GetBackendModel(backend_type));
}

// TODO: do PrepareUnitSubgraphScheduling after this
BandStatus ModelAnalyzer::GetUnitSubgraphs(
    ModelId model_id, bool need_fallback_subgraph,
    std::vector<std::pair<WorkerId, std::set<size_t>>>& unit_subgraphs) {
  const int num_workers = context_.GetNumWorkers();
  if (!need_fallback_subgraph) {
    std::set<size_t> entire_ops;
    for (int i = 0; i < model_spec_.num_ops; i++) {
      entire_ops.insert(i);
    }
    for (WorkerId id = 0; id < num_workers; id++) {
      unit_subgraphs.push_back({id, entire_ops});
    }
    return kBandOk;
  }

  const int num_ops = model_spec_.num_ops;

  using BitMask = uint64_t;
  if (num_workers > 8 * sizeof(BitMask)) {
    BAND_REPORT_ERROR(context_.GetErrorReporter(),
                      "Number of workers is larger than BitMask %d",
                      num_workers);
    return kBandError;
  }

  std::map<WorkerId, std::set<int>> op_sets_to_ignore;
  // register subgraphs for all workers
  for (int i = 0; i < num_workers; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
    std::vector<DeviceOpIndices> device_op_sets =
        MakeSubgraphsForFallbackOps(model_id, device_flag);
    for (auto device_and_ops : device_op_sets) {
      auto device = device_and_ops.first;
      auto& ops = device_and_ops.second;
      if (device == kTfLiteCPU) continue;
      if (ops.size() < minimum_subgraph_size_) {
        for (auto op : ops) {
          op_sets_to_ignore[device].insert(op);
        }
      }
    }
  }
}
}  // namespace Band