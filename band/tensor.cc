#include "band/tensor.h"

#include <string.h>

#include "band/logger.h"

namespace Band {
Tensor::Tensor(ITensor* tensor_view)
    : type_(tensor_view->GetType()),
      quantization_({kBandNoQuantization, nullptr}),
      num_bytes_(tensor_view->GetBytes()),
      dims_(tensor_view->GetDims(),
            tensor_view->GetDims() + tensor_view->GetNumDims()),
      data_(new char[tensor_view->GetBytes()]),
      name_(tensor_view->GetName()) {
  SetQuantization(tensor_view->GetQuantization());
}

Tensor::~Tensor() {
  delete[] data_;
  BandQuantizationFree(&quantization_);
}

DataType Tensor::GetType() const { return type_; }

void Tensor::SetType(DataType type) { type_ = type; }

const char* Tensor::GetData() const { return data_; }

char* Tensor::GetData() { return data_; }

const int* Tensor::GetDims() const { return dims_.data(); }

size_t Tensor::GetNumDims() const { return dims_.size(); }

void Tensor::SetDims(const std::vector<int>& dims) {
  dims_ = std::vector<int>(dims.begin(), dims.end());
}

size_t Tensor::GetBytes() const { return num_bytes_; }

const char* Tensor::GetName() const { return name_.c_str(); }

BandQuantization Tensor::GetQuantization() const { return quantization_; }

void Tensor::SetQuantization(BandQuantization quantization) {
  quantization_.type = BandQuantizationType(quantization.type);
  if (quantization_.type == kBandAffineQuantization) {
    if (quantization_.params != nullptr) {
      BandQuantizationFree(&quantization_);
    }

    const BandAffineQuantization* input_q_params =
        (BandAffineQuantization*)(quantization.params);

    BandAffineQuantization* q_params =
        reinterpret_cast<BandAffineQuantization*>(
            malloc(sizeof(BandAffineQuantization)));
    q_params->scale = BandFloatArrayCreate(input_q_params->scale->size);
    q_params->zero_point = BandIntArrayCreate(input_q_params->zero_point->size);
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