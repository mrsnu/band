#include "band/backend/tfl/tensor.h"

#include "tensorflow/lite/context_util.h"

// memcpy
#include <string.h>

namespace Band {
namespace TfLite {
TfLiteTensorView::TfLiteTensorView(TfLiteTensor* tensor) : tensor_(tensor) {}

BandBackendType TfLiteTensorView::GetBackendType() const { return kBandTfLite; }

BandType TfLiteTensorView::GetType() const { return BandType(tensor_->type); }

void TfLiteTensorView::SetType(BandType type) {
  tensor_->type = TfLiteType(type);
}

const char* TfLiteTensorView::GetData() const {
  return tensor_->data.raw_const;
}

char* TfLiteTensorView::GetData() { return tensor_->data.raw; }

std::vector<int> TfLiteTensorView::GetDims() const {
  auto view = tflite::TfLiteIntArrayView(tensor_->dims);
  return std::vector<int>(view.begin(), view.end());
}

void TfLiteTensorView::SetDims(const std::vector<int>& dims) {
  if (dims.size() == tensor_->dims->size) {
    for (int i = 0; i < dims.size(); i++) {
      tensor_->dims->data[i] = dims[i];
    }
  }
}

size_t TfLiteTensorView::GetBytes() const { return tensor_->bytes; }

const char* TfLiteTensorView::GetName() const { return tensor_->name; }

BandQuantization TfLiteTensorView::GetQuantization() const {
  BandQuantization q;
  q.params = tensor_->quantization.params;
  q.type = BandQuantizationType(tensor_->quantization.type);
  return q;
}

void TfLiteTensorView::SetQuantization(BandQuantization quantization) {
  tensor_->quantization.type = TfLiteQuantizationType(quantization.type);
  if (tensor_->quantization.type == kTfLiteAffineQuantization) {
    BandAffineQuantization* input_q_params =
        (BandAffineQuantization*)(tensor_->quantization.params);

    TfLiteAffineQuantization* q_params =
        (TfLiteAffineQuantization*)(tensor_->quantization.params);

    q_params->quantized_dimension = input_q_params->quantized_dimension;

    memcpy(
        q_params->scale, input_q_params->scale->data,
        sizeof(input_q_params->scale->data[0]) * input_q_params->scale->size);
    memcpy(q_params->zero_point, input_q_params->zero_point->data,
           sizeof(input_q_params->zero_point->data[0]) *
               input_q_params->zero_point->size);
  }
}
}  // namespace TfLite
}  // namespace Band