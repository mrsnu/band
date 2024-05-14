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

#include "band/buffer/operator.h"

#include "band/logger.h"
#include "operator.h"

namespace band {

IBufferOperator::IBufferOperator(IBufferOperator&& rhs)
    : output_(std::move(rhs.output_)), output_assigned_(rhs.output_assigned_) {}

IBufferOperator::IBufferOperator(const IBufferOperator& rhs) {
  if (rhs.output_assigned_) {
    output_ =
        std::unique_ptr<Buffer, void (*)(Buffer*)>(nullptr, [](Buffer*) {});
  } else {
    // deep copy
    output_ = std::unique_ptr<Buffer, void (*)(Buffer*)>(
        rhs.output_ ? new Buffer(*rhs.output_) : nullptr,
        [](Buffer* buffer) { delete buffer; });
  }
  output_assigned_ = rhs.output_assigned_;
}

IBufferOperator& IBufferOperator::operator=(const IBufferOperator& rhs) {
  if (rhs.output_assigned_) {
    output_ =
        std::unique_ptr<Buffer, void (*)(Buffer*)>(nullptr, [](Buffer*) {});
  } else {
    // deep copy
    output_ = std::unique_ptr<Buffer, void (*)(Buffer*)>(
        rhs.output_ ? new Buffer(*rhs.output_) : nullptr,
        [](Buffer* buffer) { delete buffer; });
  }
  output_assigned_ = rhs.output_assigned_;
  return *this;
}

absl::Status IBufferOperator::Process(const Buffer& input) {
  RETURN_IF_ERROR(ValidateInput(input));
  RETURN_IF_ERROR(ValidateOrCreateOutput(input));
  RETURN_IF_ERROR(ProcessImpl(input));
  return absl::OkStatus();
}

void IBufferOperator::SetOutput(Buffer* output) {
  if (output) {
    output_assigned_ = true;
    // no need to delete the previous output buffer
    output_ =
        std::unique_ptr<Buffer, void (*)(Buffer*)>(output, [](Buffer*) {});
  }
}

Buffer* IBufferOperator::GetOutput() { return output_.get(); }
const Buffer* IBufferOperator::GetOutput() const { return output_.get(); }

absl::Status IBufferOperator::ValidateInput(const Buffer& input) const {
  return absl::OkStatus();
}

absl::Status IBufferOperator::ValidateOrCreateOutput(const Buffer& input) {
  absl::Status status = output_ == nullptr
                            ? absl::InternalError("Null output buffer")
                            : ValidateOutput(input);
  if (!status.ok()) {
    if (output_assigned_) {
      // if output is externally assigned, return the error
      return status;
    } else {
      // if output is not externally assigned, try to create it
      RETURN_IF_ERROR(CreateOutput(input));
      // validate the created output
      status = ValidateOutput(input);
      if (!status.ok()) {
        // real trouble if the created output is not valid
        BAND_LOG(LogSeverity::kError,
                 "Failed to create valid output buffer: %s", status.ToString());
      }
      return status;
    }
  }
  return absl::OkStatus();
}
}  // namespace band
