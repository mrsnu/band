#include "band/tensor.h"

#include <string.h>

#include "band/logger.h"

namespace band {
Tensor::Tensor(ITensor* tensor_view)
    : type_(tensor_view->GetType()),
      quantization_({QuantizationType::NoQuantization, nullptr}),
      num_bytes_(tensor_view->GetBytes()),
      dims_(tensor_view->GetDims(),
            tensor_view->GetDims() + tensor_view->GetNumDims()),
      data_(new char[tensor_view->GetBytes()]),
      name_(tensor_view->GetName()) {
  auto status = SetQuantization(tensor_view->GetQuantization());
  if (!status.ok()) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set quantization: %s", status.message());
  }
}

Tensor::~Tensor() { delete[] data_; }

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

Quantization Tensor::GetQuantization() const { return quantization_; }

absl::Status Tensor::SetQuantization(Quantization quantization) {
  if (quantization_.GetType() == QuantizationType::AffineQuantization) {
    AffineQuantizationParams* input_q_params =
        reinterpret_cast<AffineQuantizationParams*>(quantization.GetParams());

    AffineQuantizationParams* q_params =
        reinterpret_cast<AffineQuantizationParams*>(
            malloc(sizeof(AffineQuantizationParams)));
    q_params->scale = FloatArray(input_q_params->scale.size());
    q_params->zero_point = IntArray(input_q_params->zero_point.size());

    {
      auto status = q_params->scale.CopyFrom(input_q_params->scale.size(),
                                             input_q_params->scale.data());
      if (!status.ok()) {
        return status;
      }
    }
    {
      auto status = q_params->zero_point.CopyFrom(input_q_params->zero_point.size(),
                                                  input_q_params->zero_point.data());
      if (!status.ok()) {
        return status;
      }
    }
                                    
    q_params->quantized_dimension = input_q_params->quantized_dimension;
  }
  return absl::OkStatus();
}

}  // namespace band