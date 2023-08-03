#include "band/backend/grpc/model_executor.h"

#include "absl/strings/str_format.h"
#include "band/backend/grpc/grpc_client.h"
#include "band/backend/grpc/model.h"
#include "band/backend/grpc/tensor.h"

namespace band {
namespace grpc {

namespace {

// TODO(widiba03304): This function only guarantees the equality of the
// structure. This function should be updated to check the equality of the
// weights, maybe with sha256sum?
bool CompareModelDesc(band_proto::ModelDescriptor lhs,
                      band_proto::ModelDescriptor rhs) {
  if (lhs.id() != rhs.id()) {
    return false;
  }
  if (lhs.num_ops() != rhs.num_ops()) {
    return false;
  }
  if (lhs.num_tensors() != rhs.num_tensors()) {
    return false;
  }
  if (lhs.input_tensor_indices_size() != rhs.input_tensor_indices_size()) {
    return false;
  }
  if (lhs.output_tensor_indices_size() != rhs.output_tensor_indices_size()) {
    return false;
  }
  if (lhs.op_input_tensors_size() != rhs.op_input_tensors_size()) {
    return false;
  }
  if (lhs.op_output_tensors_size() != rhs.op_output_tensors_size()) {
    return false;
  }
  for (int i = 0; i < lhs.input_tensor_indices_size(); i++) {
    if (lhs.input_tensor_indices(i) != rhs.input_tensor_indices(i)) {
      return false;
    }
  }
  for (int i = 0; i < lhs.output_tensor_indices_size(); i++) {
    if (lhs.output_tensor_indices(i) != rhs.output_tensor_indices(i)) {
      return false;
    }
  }
  for (int i = 0; i < lhs.op_input_tensors_size(); i++) {
    if (lhs.op_input_tensors(i).op().size() !=
        rhs.op_input_tensors(i).op().size()) {
      return false;
    }
    for (int j = 0; j < lhs.op_input_tensors(i).op().size(); j++) {
      if (lhs.op_input_tensors(i).op(j) != rhs.op_input_tensors(i).op(j)) {
        return false;
      }
    }
  }
  for (int i = 0; i < lhs.op_output_tensors_size(); i++) {
    if (lhs.op_output_tensors(i).op().size() !=
        rhs.op_output_tensors(i).op().size()) {
      return false;
    }
    for (int j = 0; j < lhs.op_output_tensors(i).op().size(); j++) {
      if (lhs.op_output_tensors(i).op(j) != rhs.op_output_tensors(i).op(j)) {
        return false;
      }
    }
  }
  return true;
}

}  // anonymous namespace

GrpcModelExecutor::~GrpcModelExecutor() {}

absl::StatusOr<ModelSpec> GrpcModelExecutor::InvestigateModelSpec(
    interface::IModel* model) {
  // 0. Request to the server to get model spec.
  auto status_or_model_descs = client_.GetModelDesc();
  if (!status_or_model_descs.ok()) {
    return status_or_model_descs.status();
  }
  auto remote_model_descs = status_or_model_descs.value();

  // 1. Get from model descriptor file.
  auto local_model = reinterpret_cast<GrpcModel*>(model);
  auto status_or_model_desc = local_model->ToProto();
  if (!status_or_model_desc.ok()) {
    return status_or_model_desc.status();
  }
  auto local_model_desc = status_or_model_desc.value();

  // 2. Compare the model descriptor with the remote ones.
  bool found = false;
  for (auto& remote_model_desc : remote_model_descs) {
    if (remote_model_desc.id() == local_model_desc.id()) {
      found = true;
      if (CompareModelDesc(remote_model_desc, local_model_desc)) {
        return absl::InternalError(
            "The model descriptor is not matched with the cloud.");
      }
    }
  }
  if (!found) {
    return absl::InternalError("No such model registered in the cloud.");
  }

  return ModelSpec(
      local_model->num_ops, local_model->num_tensors, local_model->tensor_types,
      std::set<int>(local_model->input_tensor_indices.begin(),
                    local_model->input_tensor_indices.end()),
      std::set<int>(local_model->output_tensor_indices.begin(),
                    local_model->output_tensor_indices.end()),
      local_model->op_input_tensors, local_model->op_output_tensors, {},
      {DeviceFlag::kCPU, DeviceFlag::kGPU, DeviceFlag::kDSP, DeviceFlag::kNPU});
}

absl::Status GrpcModelExecutor::PrepareSubgraph(interface::IModel* model,
                                                std::set<int> ops,
                                                std::set<int> unit_indices) {
  if (model_id_ != model->GetId()) {
    return absl::InternalError(
        absl::StrFormat("Failed to prepare subgraph, given model id %d != "
                        "predeclared interpreter's model id %d",
                        model->GetId(), model_id_));
  }
  auto local_model = reinterpret_cast<GrpcModel*>(model);
  model_descriptors_[SubgraphKey(model->GetId(), worker_id_, unit_indices)] =
      local_model;
  return absl::OkStatus();
}

BackendType GrpcModelExecutor::GetBackendType() const {
  return BackendType::kGrpc;
}

const std::vector<int>& GrpcModelExecutor::GetInputs(
    const SubgraphKey& key) const {
  return model_descriptors_.at(key)->input_tensor_indices;
}

const std::vector<int>& GrpcModelExecutor::GetOutputs(
    const SubgraphKey& key) const {
  return model_descriptors_.at(key)->output_tensor_indices;
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
  return model_descriptors_.at(key)->num_tensors;
}

size_t GrpcModelExecutor::GetNumNodes(const SubgraphKey& key) const {
  return model_descriptors_.at(key)->num_ops;
}

std::shared_ptr<interface::ITensorView> GrpcModelExecutor::GetTensorView(
    const SubgraphKey& key, int index) {
  // TODO(widiba03304): Request to the server to get intermediate tensors.
  return nullptr;
}

SubgraphKey GrpcModelExecutor::GetLargestSubgraphKey() const {
  return SubgraphKey();
}

bool GrpcModelExecutor::HasSubgraph(const SubgraphKey& key) const {
  return model_descriptors_.find(key) != model_descriptors_.end();
}

absl::Status GrpcModelExecutor::ExecuteSubgraph(const SubgraphKey& key) {
  if (!HasSubgraph(key)) {
    return absl::InternalError("Cannot find subgraph");
  }

  // TODO(widiba03304): Request to the server.
  return absl::OkStatus();
}

void GrpcModelExecutor::ForEachSubgraph(
    std::function<void(const SubgraphKey&)> visitor) {
  for (const auto& model_descriptor_ : model_descriptors_) {
    visitor(model_descriptor_.first);
  }
}

}  // namespace grpc
}  // namespace band