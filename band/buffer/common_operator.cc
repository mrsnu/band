#include "common_operator.h"

#include "absl/strings/str_format.h"
#include "band/buffer/common_operator.h"
#include "band/logger.h"

namespace band {
namespace buffer {

IBufferOperator* Normalize::Clone() const { return new Normalize(*this); }

IBufferOperator::Type Normalize::GetOpType() const { return Type::kCommon; }

void Normalize::SetOutput(Buffer* output) {
  if (inplace_) {
    BAND_LOG_PROD(
        BAND_LOG_ERROR,
        "Normalize: setting output buffer is not allowed for inplace");
  } else {
    IBufferOperator::SetOutput(output);
  }
}

absl::Status Normalize::ProcessImpl(const Buffer& input) {
  Buffer* output = inplace_ ? const_cast<Buffer*>(&input) : GetOutput();

  switch (input.GetDataType()) {
    case DataType::kUInt8:
      NormalizeImpl<uint8_t>(input, output);
      break;
    case DataType::kInt8:
      NormalizeImpl<int8_t>(input, output);
      break;
    case DataType::kInt16:
      NormalizeImpl<int16_t>(input, output);
      break;
    case DataType::kInt32:
      NormalizeImpl<int32_t>(input, output);
      break;
    case DataType::kFloat32:
      NormalizeImpl<float>(input, output);
      break;
    case DataType::kFloat64:
      NormalizeImpl<double>(input, output);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("data type %d is not supported.",
                          static_cast<int>(input.GetDataType())));
  }

  return absl::Status();
}

absl::Status Normalize::ValidateInput(const Buffer& input) const {
  if (Buffer::IsYUV(input.GetBufferFormat())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input buffer format %d is not supported.",
                        static_cast<int>(input.GetBufferFormat())));
  }

  if (input.GetNumPlanes() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input buffer should have only one plane, but it has "
                        "%d planes.",
                        input.GetNumPlanes()));
  }

  return absl::Status();
}

absl::Status Normalize::ValidateOutput(const Buffer& input) const {
  if (inplace_) {
    return absl::OkStatus();
  } else {
    if (input.GetBufferFormat() != output_->GetBufferFormat()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("input buffer should have the same format as output "
                          "buffer, but input format is %d and output format is "
                          "%d.",
                          static_cast<int>(input.GetBufferFormat()),
                          static_cast<int>(output_->GetBufferFormat())));
    }

    if (input.GetDimension() != output_->GetDimension()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "input buffer should have the same dimension as "
          "output buffer, but input dimension is %d x %d and "
          "output dimension is %d x %d.",
          input.GetDimension()[0], input.GetDimension()[1],
          output_->GetDimension()[0], output_->GetDimension()[1]));
    }
  }
  return absl::Status();
}

absl::Status Normalize::CreateOutput(const Buffer& input) {
  if (!inplace_) {
    output_ =
        Buffer::CreateEmpty(input.GetDimension()[0], input.GetDimension()[1],
                            input.GetBufferFormat(), input.GetOrientation());
  }

  return absl::Status();
}

}  // namespace buffer
}  // namespace band