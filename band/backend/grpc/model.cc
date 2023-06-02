#include "band/backend/grpc/model.h"

#include <fstream>

#include "band/backend/grpc/proto/model.pb.h"

namespace band {
namespace grpc {

GrpcModel::GrpcModel(ModelId id) : interface::IModel(id) {}

BackendType GrpcModel::GetBackendType() const { return BackendType::Grpc; }

absl::Status GrpcModel::FromPath(const char* filename) {
  std::ifstream fin;
  fin.open(filename, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    return absl::InternalError("Cannot open the model file.");
  }
  char* binary_buffer = nullptr;
  fin.read(binary_buffer, fin.gcount());
  return FromBuffer(binary_buffer, fin.gcount());
}

absl::Status GrpcModel::FromBuffer(const char* buffer, size_t buffer_size) {
  band_proto::ModelDescriptor model_desc_proto;
  if (!model_desc_proto.ParseFromArray(buffer, buffer_size)) {
    return absl::InternalError("Cannot parse the model descriptor file.");
  }
  
  id = model_desc_proto.id();
  num_ops = model_desc_proto.num_ops();
  num_tensors = model_desc_proto.num_tensors();
  for (int i = 0; i < model_desc_proto.tensor_types_size(); i++) {
    tensor_types.push_back(
        static_cast<DataType>(model_desc_proto.tensor_types(i)));
  }
  for (int i = 0; i < model_desc_proto.input_tensor_indices_size(); i++) {
    input_tensor_indices.push_back(model_desc_proto.input_tensor_indices(i));
  }
  for (int i = 0; i < model_desc_proto.output_tensor_indices_size(); i++) {
    output_tensor_indices.push_back(model_desc_proto.output_tensor_indices(i));
  }
  for (int i = 0; i < model_desc_proto.op_input_tensors_size(); i++) {
    std::set<int> op_input_tensor;
    for (int j = 0; j < model_desc_proto.op_input_tensors(i).op_size(); j++) {
      op_input_tensor.insert(
          static_cast<int>(model_desc_proto.op_input_tensors(i).op(j)));
    }
    op_input_tensors.push_back(op_input_tensor);
  }
  for (int i = 0; i < model_desc_proto.op_output_tensors_size(); i++) {
    std::set<int> op_output_tensor;
    for (int j = 0; j < model_desc_proto.op_output_tensors(i).op_size(); j++) {
      op_output_tensor.insert(model_desc_proto.op_output_tensors(i).op(j));
    }
    op_output_tensors.push_back(op_output_tensor);
  }
  return absl::OkStatus();
}

bool GrpcModel::IsInitialized() const {
  return id != "" && num_ops != -1 && num_tensors != -1 &&
         tensor_types.size() != 0 && input_tensor_indices.size() != 0 &&
         output_tensor_indices.size() != 0 && op_input_tensors.size() != 0 &&
         op_output_tensors.size() != 0;
}

}  // namespace grpc
}  // namespace band
