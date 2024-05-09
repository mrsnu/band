// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/tensor.h"

#include <string.h>

#include "band/logger.h"

namespace band {
Tensor::Tensor(ITensor* tensor_view, bool copy_data)
    : type_(tensor_view->GetType()),
      quantization_({QuantizationType::kNoQuantization, nullptr}),
      dims_(tensor_view->GetDims(),
            tensor_view->GetDims() + tensor_view->GetNumDims()),
      data_(new char[tensor_view->GetBytes()]),
      name_(tensor_view->GetName()) {
  auto status = SetQuantization(tensor_view->GetQuantization());
  if (!status.ok()) {
    BAND_LOG(LogSeverity::kError, "Failed to set quantization: %s",
                  std::string(status.message()).c_str());
  }
  if (copy_data) {
    memcpy(data_, tensor_view->GetData(), tensor_view->GetBytes());
  }
}

Tensor::~Tensor() {
  delete[] data_;
  if (quantization_.GetParams() != nullptr) {
    free(quantization_.GetParams());
  }
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

const char* Tensor::GetName() const { return name_.c_str(); }

Quantization Tensor::GetQuantization() const { return quantization_; }

absl::Status Tensor::SetQuantization(Quantization quantization) {
  if (quantization_.GetType() == QuantizationType::kAffineQuantization) {
    AffineQuantizationParams* input_q_params =
        reinterpret_cast<AffineQuantizationParams*>(quantization.GetParams());

    AffineQuantizationParams* q_params =
        reinterpret_cast<AffineQuantizationParams*>(
            malloc(sizeof(AffineQuantizationParams)));
    if (input_q_params == nullptr || q_params == nullptr) {
      return absl::InternalError(
          "Failed to allocate memory for quantization params");
    }

    q_params->scale = std::vector<float>(input_q_params->scale.size());
    q_params->zero_point = std::vector<int>(input_q_params->zero_point.size());

    q_params->scale.insert(q_params->scale.end(), input_q_params->scale.begin(),
                           input_q_params->scale.end());
    q_params->zero_point.insert(q_params->zero_point.end(),
                                input_q_params->zero_point.begin(),
                                input_q_params->zero_point.end());
    q_params->quantized_dimension = input_q_params->quantized_dimension;
    quantization_.SetParams(q_params);
  }
  return absl::OkStatus();
}

}  // namespace band