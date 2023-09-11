/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_BACKEND_TFL_TENSOR_H_
#define BAND_BACKEND_TFL_TENSOR_H_

#include "band/interface/tensor_view.h"
#include "tensorflow/lite/c/common.h"

namespace band {
namespace tfl {
class TfLiteTensorView : public interface::ITensorView {
 public:
  TfLiteTensorView(TfLiteTensor* tensor);

  BackendType GetBackendType() const override;
  DataType GetType() const override;
  void SetType(DataType type) override;
  const char* GetData() const override;
  char* GetData() override;
  const int* GetDims() const override;
  size_t GetNumDims() const override;
  void SetDims(const std::vector<int>& dims) override;
  size_t GetBytes() const override;
  const char* GetName() const override;
  Quantization GetQuantization() const override;
  absl::Status SetQuantization(Quantization quantization) override;

 private:
  TfLiteTensor* tensor_ = nullptr;
};
}  // namespace tfl
}  // namespace band

#endif