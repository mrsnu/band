#include "band/buffer/image_operation.h"

#include "absl/strings/str_format.h"
#include "band/buffer/libyuv_operation.h"
#include "band/buffer/operation.h"

namespace band {

absl::Status CropOperation::Process(const Buffer& input) {
  if (!output_) {
    const std::vector<size_t> crop_dimension =
        Buffer::GetCropDimension(x0_, x1_, y0_, y1_);
    output_ =
        Buffer::CreateEmpty(crop_dimension[0], crop_dimension[1],
                            input.GetBufferFormat(), input.GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Crop(input, x0_, y0_, x1_, y1_, *GetOutput());
}

absl::Status CropOperation::IsValid(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::Raw) {
    return absl::InvalidArgumentError(
        "CropOperation: Raw buffer format type is not supported.");
  }

  if (x0_ < 0 || y0_ < 0 || x1_ < 0 || y1_ < 0) {
    return absl::InvalidArgumentError(
        "CropOperation: negative crop region is not allowed.");
  }

  if (x0_ >= x1_ || y0_ >= y1_) {
    return absl::InvalidArgumentError(
        "CropOperation: invalid crop region is not allowed.");
  }

  if (x1_ > input.GetDimension()[0] || y1_ > input.GetDimension()[1]) {
    return absl::InvalidArgumentError(
        "CropOperation: crop region is out of bounds.");
  }

  if (output_ && !input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CropOperation: output buffer format type is not "
        "compatible. %s vs %s",
        GetName(input.GetBufferFormat()), GetName(output_->GetBufferFormat())));
  }

  return absl::OkStatus();
}  // namespace buffer

absl::Status ResizeOperation::Process(const Buffer& input) {
  if (!output_) {
    output_ = Buffer::CreateEmpty(dims_[0], dims_[1], input.GetBufferFormat(),
                                  input.GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Resize(input, *GetOutput());
}

absl::Status ResizeOperation::IsValid(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::Raw) {
    return absl::InvalidArgumentError(
        "ResizeOperation: Raw buffer format type is not supported.");
  }

  if (dims_.size() != 2) {
    return absl::InvalidArgumentError(
        "ResizeOperation: invalid dimension size.");
  }

  if (dims_[0] <= 0 || dims_[1] <= 0) {
    return absl::InvalidArgumentError(
        "ResizeOperation: invalid dimension value.");
  }

  if (output_) {
    switch (input.GetBufferFormat()) {
      case BufferFormat::GrayScale:
      case BufferFormat::RGB:
      case BufferFormat::NV12:
      case BufferFormat::NV21:
      case BufferFormat::YV12:
      case BufferFormat::YV21:
        if (input.GetBufferFormat() != output_->GetBufferFormat()) {
          return absl::InvalidArgumentError(
              "ResizeOperation: output buffer format type is not compatible.");
        }
        break;
      case BufferFormat::RGBA:
        if (output_->GetBufferFormat() != BufferFormat::RGB &&
            output_->GetBufferFormat() != BufferFormat::RGBA) {
          return absl::InvalidArgumentError(
              "ResizeOperation: output buffer format type is not compatible.");
        }
        break;
      default:
        return absl::InternalError(
            absl::StrFormat("Unsupported buffer format: %s.",
                            GetName(input.GetBufferFormat())));
    }
  }

  return absl::OkStatus();
}

absl::Status RotateOperation::Process(const Buffer& input) {
  if (!output_) {
    const bool is_dimension_change = (angle_deg_ / 90) % 2 == 1;
    const size_t width =
        is_dimension_change ? input.GetDimension()[1] : input.GetDimension()[0];
    const size_t height =
        is_dimension_change ? input.GetDimension()[0] : input.GetDimension()[1];

    output_ = Buffer::CreateEmpty(width, height, input.GetBufferFormat(),
                                  input.GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Rotate(input, angle_deg_, *GetOutput());
}

absl::Status RotateOperation::IsValid(const Buffer& input) const {
  if (output_) {
    if (input.GetBufferFormat() == BufferFormat::Raw) {
      return absl::InvalidArgumentError(
          "RotateOperation: Raw buffer format type is not supported.");
    }

    if (!input.IsBufferFormatCompatible(*output_)) {
      return absl::InvalidArgumentError(
          "RotateOperation: output buffer format type is not compatible.");
    }

    const bool is_dimension_change = (angle_deg_ / 90) % 2 == 1;
    const bool are_dimensions_rotated =
        (input.GetDimension()[0] == output_->GetDimension()[1]) &&
        (input.GetDimension()[1] == output_->GetDimension()[0]);
    const bool are_dimensions_equal =
        input.GetDimension() == output_->GetDimension();

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

absl::Status FlipOperation::Process(const Buffer& input) {
  if (!output_) {
    output_ =
        Buffer::CreateEmpty(input.GetDimension()[0], input.GetDimension()[1],
                            input.GetBufferFormat(), input.GetOrientation());
  }

  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return horizontal_ ? LibyuvBufferUtils::FlipHorizontally(input, *GetOutput())
                     : LibyuvBufferUtils::FlipVertically(input, *GetOutput());
}

absl::Status FlipOperation::IsValid(const Buffer& input) const {
  if (output_ && !input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(
        "FlipOperation: output buffer format type is not compatible.");
  }

  const auto& input_dims = input.GetDimension();
  const auto& output_dims = output_->GetDimension();
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] != output_dims[i]) {
      return absl::InvalidArgumentError(
          "FlipOperation: input and output buffer dimensions must be same.");
    }
  }

  return absl::OkStatus();
}

absl::Status ConvertOperation::Process(const Buffer& input) {
  absl::Status status = IsValid(input);
  if (!status.ok()) {
    return status;
  }

  return LibyuvBufferUtils::Convert(input, *GetOutput());
}

absl::Status ConvertOperation::IsValid(const Buffer& input) const {
  if (output_) {
    if (input.GetBufferFormat() == output_->GetBufferFormat()) {
      return absl::InvalidArgumentError("Formats must be different.");
    }

    switch (input.GetBufferFormat()) {
      case BufferFormat::GrayScale:
        return absl::InvalidArgumentError(
            "Grayscale format does not convert to other formats.");
      case BufferFormat::RGB:
        if (output_->GetBufferFormat() == BufferFormat::RGBA) {
          return absl::InvalidArgumentError(
              "RGB format does not convert to RGBA");
        }
        return absl::OkStatus();
      case BufferFormat::RGBA:
      case BufferFormat::NV12:
      case BufferFormat::NV21:
      case BufferFormat::YV12:
      case BufferFormat::YV21:
        return absl::OkStatus();
      default:
        return absl::InternalError(
            absl::StrFormat("Unsupported buffer format: %s.",
                            GetName(input.GetBufferFormat())));
    }

  } else {
    return absl::InvalidArgumentError(
        "ConvertOperation: output buffer is not set.");
  }
}

}  // namespace band