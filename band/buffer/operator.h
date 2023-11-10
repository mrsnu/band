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

#ifndef BAND_BUFFER_OPERATOR_H_
#define BAND_BUFFER_OPERATOR_H_

#include "absl/status/status.h"
#include "band/buffer/buffer.h"

namespace band {
// Interface for buffer operators such as crop, resize, rotate, flip, convert
// format, etc. Each operator should be able to validate an input buffer and
// process the input buffer to generate the output buffer.
// The output buffer can be explicitly assigned by calling SetOutput() or
// automatically created by the operator. Each operator should create
// output buffer if it is not explicitly assigned and cache the output buffer
// for future use (e.g. for the next operator with the same input format).
class IBufferOperator {
 public:
  IBufferOperator() : output_(nullptr, [](Buffer*) {}) {}
  virtual ~IBufferOperator() = default;

  IBufferOperator(IBufferOperator&&);
  IBufferOperator(const IBufferOperator& rhs);
  IBufferOperator& operator=(const IBufferOperator&);

  absl::Status Process(const Buffer& input);
  virtual void SetOutput(Buffer* output);
  Buffer* GetOutput();
  const Buffer* GetOutput() const;

  enum class Type {
    kImage,
    kCommon,
  };
  virtual Type GetOpType() const = 0;
  // Validate the input buffer. Return OK if the input buffer is valid for the
  // operation, otherwise return an error status.
  virtual absl::Status ValidateInput(const Buffer& input) const;
  // Validate the input and output buffers. Return OK if the input and output
  // buffers are valid for the operation, otherwise return an error status.
  absl::Status ValidateOrCreateOutput(const Buffer& input);
  // explicitly assign output buffer, otherwise it will be created automatically
  // Validate the output buffer. Return OK if the output buffer is valid for the
  // operation, otherwise return an error status.
  // This function is called only when the output buffer is not null.
  virtual absl::Status ValidateOutput(const Buffer& input) const = 0;
  // Create the output buffer. Return OK if the output buffer is created
  // successfully, otherwise return an error status.
  virtual absl::Status CreateOutput(const Buffer& input) = 0;

 protected:
  friend class ImageProcessorBuilder;
  // Process the input buffer and generate the output buffer.
  // Returns following status:
  // - OK: the operation is successful.
  // - other error status: the operation is failed.
  virtual absl::Status ProcessImpl(const Buffer& input) = 0;
  // For processor builder to clone the operator
  virtual IBufferOperator* Clone() const = 0;

  bool output_assigned_ = false;
  std::unique_ptr<Buffer, void (*)(Buffer*)> output_ =
      std::unique_ptr<Buffer, void (*)(Buffer*)>(nullptr, [](Buffer*) {});
};
}  // namespace band

#endif