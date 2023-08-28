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

#include "common_operator.h"

#include "absl/strings/str_format.h"
#include "band/buffer/common_operator.h"
#include "band/logger.h"

namespace band {
namespace buffer {

template <typename InputType, typename OutputType>
void NormalizeFromTo(const Buffer& input, Buffer* output, float mean,
                     float std) {
  // only single plane is supported
  const InputType* input_data =
      reinterpret_cast<const InputType*>(input[0].data);
  OutputType* output_data =
      reinterpret_cast<OutputType*>((*output)[0].GetMutableData());
  for (int i = 0; i < input.GetNumElements(); ++i) {
    output_data[i] = static_cast<OutputType>((input_data[i] - mean)) / std;
  }
  BAND_LOG_PROD(BAND_LOG_INFO, "Normalize: %s %s %f %f",
                ToString(input.GetDataType()), ToString(output->GetDataType()),
                mean, std);
}

template <typename InputType>
void NormalizeFrom(const Buffer& input, Buffer* output, float mean, float std) {
  switch (output->GetDataType()) {
    case DataType::kUInt8:
      NormalizeFromTo<InputType, uint8_t>(input, output, mean, std);
      break;
    case DataType::kInt8:
      NormalizeFromTo<InputType, int8_t>(input, output, mean, std);
      break;
    case DataType::kInt16:
      NormalizeFromTo<InputType, int16_t>(input, output, mean, std);
      break;
    case DataType::kInt32:
      NormalizeFromTo<InputType, int32_t>(input, output, mean, std);
      break;
    case DataType::kFloat32:
      NormalizeFromTo<InputType, float>(input, output, mean, std);
      break;
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Normalize: unsupported data type %s",
                    ToString(output->GetDataType()));
      break;
  }
}

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
      NormalizeFrom<uint8_t>(input, output, mean_, std_);
      break;
    case DataType::kInt8:
      NormalizeFrom<int8_t>(input, output, mean_, std_);
      break;
    case DataType::kInt16:
      NormalizeFrom<int16_t>(input, output, mean_, std_);
      break;
    case DataType::kInt32:
      NormalizeFrom<int32_t>(input, output, mean_, std_);
      break;
    case DataType::kFloat32:
      NormalizeFrom<float>(input, output, mean_, std_);
      break;
    case DataType::kFloat64:
      NormalizeFrom<double>(input, output, mean_, std_);
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

IBufferOperator* DataTypeConvert::Clone() const {
  return new DataTypeConvert(*this);
}

absl::Status DataTypeConvert::ProcessImpl(const Buffer& input) {
  if (input.GetDataType() != output_->GetDataType()) {
    return Normalize::ProcessImpl(input);
  }
  // do nothing if input data type is the same as output data type
  return absl::Status();
}

}  // namespace buffer
}  // namespace band