#include "band/buffer/common_operation.h"
#include "common_operation.h"

namespace band {
namespace buffer {

IBufferOperator* band::Normalize::Clone() const { return nullptr; }

IBufferOperator::Type Normalize::GetOpType() const { return Type(); }

absl::Status Normalize::ProcessImpl(const Buffer& input) {
  return absl::Status();
}

absl::Status Normalize::ValidateInput(const Buffer& input) const {
  if (IsYUV(input.GetBufferFormat())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input buffer format %d is not supported.",
                        static_cast<int>(input.GetBufferFormat())));
  }

  return absl::Status();
}

absl::Status Normalize::ValidateOutput(const Buffer& input) const {
  return absl::Status();
}

absl::Status Normalize::CreateOutput(const Buffer& input) {
  return absl::Status();
}

}  // namespace buffer
}  // namespace band