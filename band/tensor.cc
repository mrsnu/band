#include "band/tensor.h"

#include <string.h>

namespace Band {
Tensor::Tensor(ITensor* tensor_view)
    : type_(tensor_view->GetType()),
      num_bytes_(tensor_view->GetBytes()),
      dims_(tensor_view->GetDims()),
      data_(new char[tensor_view->GetBytes()]),
      name_(tensor_view->GetName()) {
  SetQuantization(tensor_view->GetQuantization());
}

Tensor::~Tensor() { delete[] data_; }

BandType Tensor::GetType() const { return type_; }

void Tensor::SetType(BandType type) { type_ = type; }

const char* Tensor::GetData() const { return data_; }

char* Tensor::GetData() { return data_; }

std::vector<int> Tensor::GetDims() const { return dims_; }

void Tensor::SetDims(const std::vector<int>& dims) {
  dims_ = std::vector<int>(dims.begin(), dims.end());
}

size_t Tensor::GetBytes() const { return num_bytes_; }

const char* Tensor::GetName() const { return name_.c_str(); }

BandQuantization Tensor::GetQuantization() const { return quantization_; }

void Tensor::SetQuantization(BandQuantization quantization) {
  quantization_.type = BandQuantizationType(quantization.type);
  if (quantization_.type == kBandAffineQuantization) {
    BandAffineQuantization* input_q_params =
        (BandAffineQuantization*)(quantization_.params);

    BandAffineQuantization* q_params =
        (BandAffineQuantization*)(quantization_.params);

    q_params->quantized_dimension = input_q_params->quantized_dimension;

    memcpy(
        q_params->scale, input_q_params->scale->data,
        sizeof(input_q_params->scale->data[0]) * input_q_params->scale->size);
    memcpy(q_params->zero_point, input_q_params->zero_point->data,
           sizeof(input_q_params->zero_point->data[0]) *
               input_q_params->zero_point->size);
  }
}

}  // namespace Band