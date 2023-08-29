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

#ifndef BAND_INTERFACE_TENSOR_H_
#define BAND_INTERFACE_TENSOR_H_

#include <vector>

#include "absl/status/status.h"
#include "band/common.h"

namespace band {
namespace interface {
struct ITensor {
 public:
  virtual ~ITensor() = default;

  virtual DataType GetType() const = 0;
  virtual void SetType(DataType type) = 0;
  virtual const char* GetData() const = 0;
  virtual char* GetData() = 0;
  virtual const int* GetDims() const = 0;
  virtual size_t GetNumDims() const = 0;
  virtual void SetDims(const std::vector<int>& dims) = 0;
  virtual const char* GetName() const = 0;
  virtual Quantization GetQuantization() const = 0;
  virtual absl::Status SetQuantization(Quantization quantization) = 0;
  bool operator==(const ITensor& rhs) const;
  bool operator!=(const ITensor& rhs) const;

  virtual size_t GetBytes() const;
  size_t GetNumElements() const;
  std::vector<int> GetDimsVector() const;

  absl::Status CopyDataFrom(const ITensor& rhs);
  absl::Status CopyDataFrom(const ITensor* rhs);
};
}  // namespace interface
}  // namespace band

#endif