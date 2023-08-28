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

#ifndef BAND_BUFFER_COMMON_OPERATION_H_
#define BAND_BUFFER_COMMON_OPERATION_H_

#include "absl/status/status.h"
#include "band/buffer/buffer.h"
#include "band/buffer/operator.h"

namespace band {
namespace buffer {

class Normalize : public IBufferOperator {
 public:
  Normalize(float mean, float std, bool inplace)
      : mean_(mean), std_(std), inplace_(inplace) {}
  virtual Type GetOpType() const override;

  virtual void SetOutput(Buffer* output) override;
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

 private:
  virtual IBufferOperator* Clone() const override;

  float mean_, std_;
  bool inplace_;
};

// DataTypeConvert is equivalent to Normalize(0.f, 1.f) without inplace.
// It automatically converts the internal data type to output data type.
class DataTypeConvert : public Normalize {
 public:
  DataTypeConvert() : Normalize(0.f, 1.f, false) {}

 private:
  virtual IBufferOperator* Clone() const override;
  virtual absl::Status ProcessImpl(const Buffer& input) override;
};

}  // namespace buffer
}  // namespace band

#endif  // BAND_BUFFER_COMMON_OPERATION_H_