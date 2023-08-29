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

#ifndef BAND_TENSOR_H_
#define BAND_TENSOR_H_

#include <string>
#include <vector>

#include "band/interface/tensor.h"

namespace band {
/*
  Tensor interface that tensor view / band tensor shares
*/
class Tensor : public interface::ITensor {
 public:
  explicit Tensor(interface::ITensor* tensor_view, bool copy_data = false);
  ~Tensor();

  DataType GetType() const override;
  void SetType(DataType type) override;
  const char* GetData() const override;
  char* GetData() override;
  const int* GetDims() const override;
  size_t GetNumDims() const override;
  void SetDims(const std::vector<int>& dims) override;
  const char* GetName() const override;
  Quantization GetQuantization() const override;
  absl::Status SetQuantization(Quantization quantization) override;

 private:
  DataType type_;
  Quantization quantization_;
  std::vector<int> dims_;
  char* data_;
  std::string name_;
};
}  // namespace band

#endif  // BAND_TENSOR_H_
