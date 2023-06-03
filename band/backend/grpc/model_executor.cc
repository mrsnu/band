#include "band/backend/grpc/model_executor.h"

#include "band/backend/grpc/proto/model.grpc.pb.h"

namespace band {
namespace grpc {

GrpcModelExecutor::~GrpcModelExecutor() {}

absl::StatusOr<ModelSpec> GrpcModelExecutor::InvestigateModelSpec(
    interface::IModel* model) {
  // Request to the server to get model spec.
  int num_ops = 0;
  int num_tensors = 0;
  std::vector<DataType> tensor_types;
  std::set<int> input_tensor_indices;
  std::set<int> output_tensor_indices;
  std::vector<std::set<int>> op_input_tensors;
  std::vector<std::set<int>> op_output_tensors;
  std::map<DeviceFlags, std::set<int>> unsupported_ops;
  std::set<DeviceFlags> unavailable_devices;

  // 1. Get from model descriptor file.
  // 2. Compare the model descriptor with the cloud.
  
  ModelSpec model_spec(num_ops, 
                       num_tensors, 
                       tensor_types, 
                       input_tensor_indices,
                       output_tensor_indices, 
                       op_input_tensors,
                       op_output_tensors, 
                       unsupported_ops, 
                       unavailable_devices);

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
    const SubgraphKey& key) const {
  return inputs_;
}

const std::vector<int>& GrpcModelExecutor::GetOutputs(
    const SubgraphKey& key) const {
  return outputs_;
}

const char* GrpcModelExecutor::GetInputName(const SubgraphKey& key,
                                            int index) const {
  return "";
}

const char* GrpcModelExecutor::GetOutputName(const SubgraphKey& key,
                                             int index) const {
  return "";
}

size_t GrpcModelExecutor::GetNumTensors(const SubgraphKey& key) const {
  return 0;
}

size_t GrpcModelExecutor::GetNumNodes(const SubgraphKey& key) const {
  return 0;
}

std::shared_ptr<interface::ITensorView> GrpcModelExecutor::GetTensorView(
    const SubgraphKey& key, int index) {
  return nullptr;
}

SubgraphKey GrpcModelExecutor::GetLargestSubgraphKey() const {
  return SubgraphKey();
}

bool GrpcModelExecutor::HasSubgraph(const SubgraphKey& key) const {
  return false;
}

absl::Status GrpcModelExecutor::ExecuteSubgraph(const SubgraphKey& key) {
  return absl::OkStatus();
}

void GrpcModelExecutor::ForEachSubgraph(
    std::function<void(const SubgraphKey&)> iterator) {
  return;
}

}  // namespace grpc
}  // namespace band