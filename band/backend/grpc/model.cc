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
  band_proto::ModelDescriptor model_desc_proto;
  if (!model_desc_proto.ParseFromIstream(&fin)) {
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
      op_input_tensor.insert(model_desc_proto.op_input_tensors(i).op(j));
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

absl::Status GrpcModel::FromBuffer(const char* buffer, size_t buffer_size) {
  band_proto::ModelDescriptor model_desc_proto;
  if (!model_desc_proto.ParseFromArray(buffer, buffer_size)) {
    return absl::InternalError("Cannot parse the model descriptor buffer.");
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
      op_input_tensor.insert(model_desc_proto.op_input_tensors(i).op(j));
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

absl::Status GrpcModel::ToPath(const char* filename) const {
  if (!IsInitialized()) {
    return absl::InternalError("Model is not initialized.");
  }

  band_proto::ModelDescriptor model_desc_proto;
  model_desc_proto.set_id(id);
  model_desc_proto.set_num_ops(num_ops);
  model_desc_proto.set_num_tensors(num_tensors);
  for (int i = 0; i < tensor_types.size(); i++) {
    model_desc_proto.add_tensor_types(
        static_cast<band_proto::DataType>(tensor_types[i]));
  }
  for (int i = 0; i < input_tensor_indices.size(); i++) {
    model_desc_proto.add_input_tensor_indices(input_tensor_indices[i]);
  }
  for (int i = 0; i < output_tensor_indices.size(); i++) {
    model_desc_proto.add_output_tensor_indices(output_tensor_indices[i]);
  }
  for (int i = 0; i < op_input_tensors.size(); i++) {
    band_proto::OpSet* op_input_tensor_proto =
        model_desc_proto.add_op_input_tensors();
    for (auto it = op_input_tensors[i].begin(); it != op_input_tensors[i].end();
         it++) {
      op_input_tensor_proto->add_op(*it);
    }
  }
  for (int i = 0; i < op_output_tensors.size(); i++) {
    band_proto::OpSet* op_output_tensor_proto =
        model_desc_proto.add_op_output_tensors();
    for (auto it = op_output_tensors[i].begin();
         it != op_output_tensors[i].end(); it++) {
      op_output_tensor_proto->add_op(*it);
    }
  }

  std::ofstream fout;
  fout.open(filename, std::ios::out | std::ios::binary);
  if (!fout.is_open()) {
    return absl::InternalError("Cannot open the model file.");
  }
  if (!model_desc_proto.SerializeToOstream(&fout)) {
    return absl::InternalError("Cannot serialize the model descriptor file.");
  }
  
  return absl::OkStatus();
}

}  // namespace grpc
}  // namespace band
