#include "band/tensor/operation.h"

#include "absl/strings/str_format.h"
#include "band/tensor/libyuv_operation.h"

namespace band {
namespace tensor {

absl::Status CropOperation::Process(const Buffer* input) {
  if (!output_.get()) {
    output_ = Buffer::CreateEmpty(x1_ - x0_, y1_ - y0_, input->GetFormatType(),
                                  input->GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Crop(*input, x0_, y0_, x1_, y1_, *GetOutput());
}

absl::Status CropOperation::IsValid(const Buffer* input) const {
  if (input->GetFormatType() == FormatType::Custom) {
    return absl::InvalidArgumentError(
        "CropOperation: Custom buffer format type is not supported.");
  }

  if (x0_ < 0 || y0_ < 0 || x1_ < 0 || y1_ < 0) {
    return absl::InvalidArgumentError(
        "CropOperation: negative crop region is not allowed.");
  }

  if (x0_ >= x1_ || y0_ >= y1_) {
    return absl::InvalidArgumentError(
        "CropOperation: invalid crop region is not allowed.");
  }

  if (x1_ > input->GetDimension()[0] || y1_ > input->GetDimension()[1]) {
    return absl::InvalidArgumentError(
        "CropOperation: crop region is out of bounds.");
  }

  if (output_.get()) {
    if (input->IsFormatTypeCompatible(*output_)) {
      return absl::InvalidArgumentError(
          "CropOperation: output buffer format type is not compatible.");
    }
  }

  return absl::OkStatus();
}

absl::Status ResizeOperation::Process(const Buffer* input) {
  if (!output_.get()) {
    output_ = Buffer::CreateEmpty(dims_[0], dims_[1], input->GetFormatType(),
                                  input->GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Resize(*input, *GetOutput());
}

absl::Status ResizeOperation::IsValid(const Buffer* input) const {
  if (input->GetFormatType() == FormatType::Custom) {
    return absl::InvalidArgumentError(
        "ResizeOperation: Custom buffer format type is not supported.");
  }

  if (dims_.size() != 2) {
    return absl::InvalidArgumentError(
        "ResizeOperation: invalid dimension size.");
  }

  if (dims_[0] <= 0 || dims_[1] <= 0) {
    return absl::InvalidArgumentError(
        "ResizeOperation: invalid dimension value.");
  }

  if (output_.get()) {
    switch (input->GetFormatType()) {
      case FormatType::GrayScale:
      case FormatType::RGB:
      case FormatType::NV12:
      case FormatType::NV21:
      case FormatType::YV12:
      case FormatType::YV21:
        if (input->GetFormatType() != output_->GetFormatType()) {
          return absl::InvalidArgumentError(
              "ResizeOperation: output buffer format type is not compatible.");
        }
        break;
      case FormatType::RGBA:
        if (output_->GetFormatType() != FormatType::RGB &&
            output_->GetFormatType() != FormatType::RGBA) {
          return absl::InvalidArgumentError(
              "ResizeOperation: output buffer format type is not compatible.");
        }
        break;
      default:
        return absl::InternalError(absl::StrFormat(
            "Unsupported buffer format: %s.", GetName(input->GetFormatType())));
    }
  }

  return absl::OkStatus();
}

absl::Status RotateOperation::Process(const Buffer* input) {
  if (!output_.get()) {
    const bool is_dimension_change = (angle_deg_ / 90) % 2 == 1;
    const size_t width = is_dimension_change ? input->GetDimension()[1]
                                             : input->GetDimension()[0];
    const size_t height = is_dimension_change ? input->GetDimension()[0]
                                              : input->GetDimension()[1];

    output_ = Buffer::CreateEmpty(width, height, input->GetFormatType(),
                                  input->GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Rotate(*input, angle_deg_, *GetOutput());
}

absl::Status RotateOperation::IsValid(const Buffer* input) const {
  if (output_.get()) {
    if (input->GetFormatType() == FormatType::Custom) {
      return absl::InvalidArgumentError(
          "RotateOperation: Custom buffer format type is not supported.");
    }

    if (input->IsFormatTypeCompatible(*output_)) {
      return absl::InvalidArgumentError(
          "RotateOperation: output buffer format type is not compatible.");
    }

    const bool is_dimension_change = (angle_deg_ / 90) % 2 == 1;
    const bool are_dimensions_rotated =
        (input->GetDimension()[0] == output_->GetDimension()[1]) &&
        (input->GetDimension()[1] == output_->GetDimension()[0]);
    const bool are_dimensions_equal =
        input->GetDimension() == output_->GetDimension();

    if (angle_deg_ >= 360 || angle_deg_ <= 0 || angle_deg_ % 90 != 0) {
      return absl::InvalidArgumentError(
          "Rotation angle must be between 0 and 360, in multiples of 90 "
          "degrees.");
    } else if ((is_dimension_change && !are_dimensions_rotated) ||
               (!is_dimension_change && !are_dimensions_equal)) {
      return absl::InvalidArgumentError(
          "Output buffer has invalid dimensions for rotation.");
    }
  }
  return absl::OkStatus();
}

absl::Status FlipOperation::Process(const Buffer* input) {
  if (!output_.get()) {
    output_ =
        Buffer::CreateEmpty(input->GetDimension()[0], input->GetDimension()[1],
                            input->GetFormatType(), input->GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return horizontal_ ? LibyuvBufferUtils::FlipHorizontally(*input, *GetOutput())
                     : LibyuvBufferUtils::FlipVertically(*input, *GetOutput());
}

absl::Status FlipOperation::IsValid(const Buffer* input) const {
  if (output_.get()) {
    if (input->IsFormatTypeCompatible(*output_)) {
      return absl::InvalidArgumentError(
          "FlipOperation: output buffer format type is not compatible.");
    }

    const auto& input_dims = input->GetDimension();
    const auto& output_dims = output_->GetDimension();
    for (size_t i = 0; i < input_dims.size(); ++i) {
      if (input_dims[i] != output_dims[i]) {
        return absl::InvalidArgumentError(
            "FlipOperation: input and output buffer dimensions must be same.");
      }
    }
  }

  return absl::OkStatus();
}

absl::Status ConvertOperation::Process(const Buffer* input) {
  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Convert(*input, *GetOutput());
}

absl::Status ConvertOperation::IsValid(const Buffer* input) const {
  if (output_.get()) {
    if (input->GetFormatType() == output_->GetFormatType()) {
      return absl::InvalidArgumentError("Formats must be different.");
    }

    switch (input->GetFormatType()) {
      case FormatType::GrayScale:
        return absl::InvalidArgumentError(
            "Grayscale format does not convert to other formats.");
      case FormatType::RGB:
        if (output_->GetFormatType() == FormatType::RGBA) {
          return absl::InvalidArgumentError(
              "RGB format does not convert to RGBA");
        }
        return absl::OkStatus();
      case FormatType::RGBA:
      case FormatType::NV12:
      case FormatType::NV21:
      case FormatType::YV12:
      case FormatType::YV21:
        return absl::OkStatus();
      default:
        return absl::InternalError(absl::StrFormat(
            "Unsupported buffer format: %s.", GetName(input->GetFormatType())));
    }

  } else {
    return absl::InvalidArgumentError(
        "ConvertOperation: output buffer is not set.");
  }
}

}  // namespace tensor
}  // namespace band