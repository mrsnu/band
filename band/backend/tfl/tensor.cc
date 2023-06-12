#include "band/backend/tfl/tensor.h"

#include "tensorflow/lite/context_util.h"

// memcpy
#include <string.h>

namespace band {
namespace tfl {
TfLiteTensorView::TfLiteTensorView(TfLiteTensor* tensor) : tensor_(tensor) {}

BackendType TfLiteTensorView::GetBackendType() const {
  return BackendType::kTfLite;
}

DataType TfLiteTensorView::GetType() const { return DataType(tensor_->type); }

void TfLiteTensorView::SetType(DataType type) {
  tensor_->type = TfLiteType(type);
}

const char* TfLiteTensorView::GetData() const {
  return tensor_->data.raw_const;
}

char* TfLiteTensorView::GetData() { return tensor_->data.raw; }

const int* TfLiteTensorView::GetDims() const { return tensor_->dims->data; }

size_t TfLiteTensorView::GetNumDims() const { return tensor_->dims->size; }

void TfLiteTensorView::SetDims(const std::vector<int>& dims) {
  if (dims.size() == static_cast<size_t>(tensor_->dims->size)) {
    for (size_t i = 0; i < dims.size(); i++) {
      tensor_->dims->data[i] = dims[i];
    }
  }
}

size_t TfLiteTensorView::GetBytes() const { return tensor_->bytes; }

const char* TfLiteTensorView::GetName() const { return tensor_->name; }

Quantization TfLiteTensorView::GetQuantization() const {
  return {QuantizationType(tensor_->quantization.type),
          tensor_->quantization.params};
}

absl::Status TfLiteTensorView::SetQuantization(Quantization quantization) {
  tensor_->quantization.type = TfLiteQuantizationType(quantization.GetType());
  switch (quantization.GetType()) {
    case QuantizationType::kAffineQuantization: {
      AffineQuantizationParams* input_q_params =
          reinterpret_cast<AffineQuantizationParams*>(
              tensor_->quantization.params);

      TfLiteAffineQuantization* q_params =
          reinterpret_cast<TfLiteAffineQuantization*>(
              tensor_->quantization.params);

      memcpy(q_params->scale->data, input_q_params->scale.data(),
             input_q_params->scale.size() * sizeof(float));
      memcpy(q_params->zero_point->data, input_q_params->zero_point.data(),
             input_q_params->zero_point.size() * sizeof(int32_t));
      q_params->quantized_dimension = input_q_params->quantized_dimension;
    } break;
    case QuantizationType::kNoQuantization:
      break;
    default:
      break;
  }
  return absl::OkStatus();
}

}  // namespace tfl
}  // namespace band