#include "band/backend/grpc/model_executor.h"

namespace band {
namespace grpc {

GrpcModelExecutor::~GrpcModelExecutor() {}

absl::StatusOr<ModelSpec> GrpcModelExecutor::InvestigateModelSpec(
    interface::IModel* model) {
  // Request to the server to get model spec.

  ModelSpec model_spec(num_ops, num_tensors, tensor_types, input_tensor_indices,
                       output_tensor_indices, op_input_tensors,
                       op_output_tensors, unsupported_ops, unavailable_devices);

  model_spec.path = model->GetPath();
  return model_spec;
}

absl::Status GrpcModelExecutor::PrepareSubgraph(interface::IModel* model,
                                                std::set<int> ops,
                                                std::set<int> unit_indices) {
  return absl::OkStatus();
}

BackendType GrpcModelExecutor::GetBackendType() const {
    return BackendType::Grpc;
}

const std::vector<int>& GrpcModelExecutor::GetInputs(
    const SubgraphKey& key) const {}

const std::vector<int>& GrpcModelExecutor::GetOutputs(
    const SubgraphKey& key) const {}

const char* GrpcModelExecutor::GetInputName(const SubgraphKey& key,
                                            int index) const {}

const char* GrpcModelExecutor::GetOutputName(const SubgraphKey& key,
                                             int index) const {}

size_t GrpcModelExecutor::GetNumTensors(const SubgraphKey& key) const {}

size_t GrpcModelExecutor::GetNumNodes(const SubgraphKey& key) const {}

std::shared_ptr<interface::ITensorView> GrpcModelExecutor::GetTensorView(
    const SubgraphKey& key, int index) {}

SubgraphKey GrpcModelExecutor::GetLargestSubgraphKey() const {}

bool GrpcModelExecutor::HasSubgraph(const SubgraphKey& key) const {}

absl::Status GrpcModelExecutor::ExecuteSubgraph(const SubgraphKey& key) {}

void GrpcModelExecutor::ForEachSubgraph(
    std::function<void(const SubgraphKey&)> iterator) {}

}  // namespace grpc
}  // namespace band