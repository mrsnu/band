#include "band/backend/grpc/tensor.h"

namespace band {
namespace grpc {

GrpcTensorView::GrpcTensorView(band_proto::Tensor& tensor) : tensor_(tensor) {}

BackendType GrpcTensorView::GetBackendType() const { return BackendType::Grpc; }

DataType GrpcTensorView::GetType() const {
  return static_cast<DataType>(tensor_.dtype());
}

void GrpcTensorView::SetType(DataType type) {
  tensor_.set_dtype(static_cast<band_proto::DataType>(type));
}

const char* GrpcTensorView::GetData() const { return tensor_.data().data(); }

char* GrpcTensorView::GetData() {
  // Note: It seems weird but it is correct. `mutable_data()` returns a
  // `string*` and to get the mutable data we need to get the pointer to the
  // first element of the string.
  return &((*(tensor_.mutable_data()))[0]);
}

const int* GrpcTensorView::GetDims() const {
  return tensor_.shape().dims().data();
}

size_t GrpcTensorView::GetNumDims() const {
  return tensor_.shape().dims().size();
}

void GrpcTensorView::SetDims(const std::vector<int>& dims) {
  tensor_.mutable_shape()->mutable_dims()->CopyFrom(
      google::protobuf::RepeatedField<int>(dims.begin(), dims.end()));
}

size_t GrpcTensorView::GetBytes() const { return tensor_.data().size(); }

const char* GrpcTensorView::GetName() const { return ""; }

Quantization GrpcTensorView::GetQuantization() const {
  Quantization quantization(
      static_cast<QuantizationType>(tensor_.quantization().type()), nullptr);
  switch (quantization.GetType()) {
    case QuantizationType::NoQuantization: {
    } break;
    case QuantizationType::AffineQuantization: {
      auto param = new AffineQuantizationParams;
      param->scale = std::vector<float>(
          tensor_.quantization().affine_param().scale().begin(),
          tensor_.quantization().affine_param().scale().end());
      param->zero_point = std::vector<int>(
          tensor_.quantization().affine_param().zero_point().begin(),
          tensor_.quantization().affine_param().zero_point().end());
      param->quantized_dimension =
          tensor_.quantization().affine_param().quantized_dimension();
      quantization.SetParams(param);
    } break;
    default: {
    }
  }
  return quantization;
}

absl::Status GrpcTensorView::SetQuantization(Quantization quantization) {
  tensor_.mutable_quantization()->set_type(
      static_cast<band_proto::QuantizationType>(quantization.GetType()));
  switch (quantization.GetType()) {
    case QuantizationType::NoQuantization: {
    } break;
    case QuantizationType::AffineQuantization: {
      auto param =
          static_cast<AffineQuantizationParams*>(quantization.GetParams());
      tensor_.mutable_quantization()
          ->mutable_affine_param()
          ->mutable_scale()
          ->CopyFrom(google::protobuf::RepeatedField<float>(
              param->scale.begin(), param->scale.end()));
      tensor_.mutable_quantization()
          ->mutable_affine_param()
          ->mutable_zero_point()
          ->CopyFrom(google::protobuf::RepeatedField<int32_t>(
              param->zero_point.begin(), param->zero_point.end()));
      tensor_.mutable_quantization()
          ->mutable_affine_param()
          ->set_quantized_dimension(param->quantized_dimension);
    } break;
    default: {
    }
  }
  return absl::OkStatus();
}

}  // namespace grpc
}  // namespace band