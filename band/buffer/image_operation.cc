#include "band/buffer/image_operation.h"

#include "absl/strings/str_format.h"
#include "band/buffer/libyuv_operation.h"
#include "band/buffer/operation.h"
#include "image_operation.h"

namespace band {
IOperation* CropOperation::Clone() const { return new CropOperation(*this); }

IOperation* ResizeOperation::Clone() const {
  return new ResizeOperation(*this);
}

IOperation* RotateOperation::Clone() const {
  return new RotateOperation(*this);
}

IOperation* FlipOperation::Clone() const { return new FlipOperation(*this); }

IOperation* ConvertOperation::Clone() const {
  return new ConvertOperation(*this);
}

IOperation::OperationType CropOperation::GetOperationType() const {
  return IOperation::OperationType::kImage;
}

IOperation::OperationType ResizeOperation::GetOperationType() const {
  return IOperation::OperationType::kImage;
}

IOperation::OperationType RotateOperation::GetOperationType() const {
  return IOperation::OperationType::kImage;
}

IOperation::OperationType FlipOperation::GetOperationType() const {
  return IOperation::OperationType::kImage;
}

IOperation::OperationType ConvertOperation::GetOperationType() const {
  return IOperation::OperationType::kImage;
}

absl::Status CropOperation::ProcessImpl(const Buffer& input) {
  return LibyuvBufferUtils::Crop(input, x0_, y0_, x1_, y1_, *GetOutput());
}

absl::Status CropOperation::ValidateInput(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::kRaw) {
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

  return absl::OkStatus();
}  // namespace buffer

absl::Status CropOperation::ValidateOutput(const Buffer& input) const {
  if (!input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("CropOperation: output buffer format type is not "
                        "compatible. %s vs %s",
                        ToString(input.GetBufferFormat()),
                        ToString(output_->GetBufferFormat())));
  }

  const std::vector<size_t> crop_dimension =
      Buffer::GetCropDimension(x0_, x1_, y0_, y1_);

  if (crop_dimension[0] != output_->GetDimension()[0] ||
      crop_dimension[1] != output_->GetDimension()[1]) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CropOperation: output buffer dimension is not "
        "compatible. %d x %d vs %d x %d",
        crop_dimension[0], crop_dimension[1], output_->GetDimension()[0],
        output_->GetDimension()[1]));
  }

  return absl::OkStatus();
}

absl::Status CropOperation::CreateOutput(const Buffer& input) {
  const std::vector<size_t> crop_dimension =
      Buffer::GetCropDimension(x0_, x1_, y0_, y1_);
  output_ =
      Buffer::CreateEmpty(crop_dimension[0], crop_dimension[1],
                          input.GetBufferFormat(), input.GetOrientation());
  return absl::OkStatus();
}

absl::Status ResizeOperation::ProcessImpl(const Buffer& input) {
  return LibyuvBufferUtils::Resize(input, *GetOutput());
}

absl::Status ResizeOperation::ValidateInput(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::kRaw) {
    return absl::InvalidArgumentError(
        "ResizeOperation: Raw buffer format type is not supported.");
  }

  if (dims_.size() < 2) {
    return absl::InvalidArgumentError(
        "ResizeOperation: invalid dimension size.");
  }

  return absl::OkStatus();
}

absl::Status ResizeOperation::ValidateOutput(const Buffer& input) const {
  switch (input.GetBufferFormat()) {
    case BufferFormat::kGrayScale:
    case BufferFormat::kRGB:
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      if (input.GetBufferFormat() != output_->GetBufferFormat()) {
        return absl::InvalidArgumentError(
            "ResizeOperation: output buffer format type is not compatible.");
      }
      break;
    case BufferFormat::kRGBA:
      if (output_->GetBufferFormat() != BufferFormat::kRGB &&
          output_->GetBufferFormat() != BufferFormat::kRGBA) {
        return absl::InvalidArgumentError(
            "ResizeOperation: output buffer format type is not compatible.");
      }
      break;
    default:
      return absl::InternalError(absl::StrFormat(
          "Unsupported buffer format: %s.", ToString(input.GetBufferFormat())));
  }

  for (size_t i = 0; i < dims_.size(); ++i) {
    if (!IsAuto(i) && dims_[i] != output_->GetDimension()[i]) {
      return absl::InvalidArgumentError(
          absl::StrFormat("ResizeOperation: output buffer dimension is not "
                          "compatible. %d != %zd",
                          dims_[i], output_->GetDimension()[i]));
    }
  }

  return absl::Status();
}

absl::Status ResizeOperation::CreateOutput(const Buffer& input) {
  if (IsAuto(0) || IsAuto(1)) {
    return absl::InvalidArgumentError(
        "ResizeOperation: cannot create output buffer with auto dimension.");
  }

  output_ = Buffer::CreateEmpty(dims_[0], dims_[1], input.GetBufferFormat(),
                                input.GetOrientation());
  return absl::OkStatus();
}

absl::Status RotateOperation::ProcessImpl(const Buffer& input) {
  return LibyuvBufferUtils::Rotate(input, angle_deg_, *GetOutput());
}

absl::Status RotateOperation::ValidateInput(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::kRaw) {
    return absl::InvalidArgumentError(
        "RotateOperation: Raw buffer format type is not supported.");
  }
  return absl::OkStatus();
}

absl::Status RotateOperation::ValidateOutput(const Buffer& input) const {
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
  return absl::Status();
}

absl::Status RotateOperation::CreateOutput(const Buffer& input) {
  const bool is_dimension_change = (angle_deg_ / 90) % 2 == 1;
  const size_t width =
      is_dimension_change ? input.GetDimension()[1] : input.GetDimension()[0];
  const size_t height =
      is_dimension_change ? input.GetDimension()[0] : input.GetDimension()[1];

  output_ = Buffer::CreateEmpty(width, height, input.GetBufferFormat(),
                                input.GetOrientation());
  return absl::OkStatus();
}

absl::Status FlipOperation::ProcessImpl(const Buffer& input) {
  return horizontal_ ? LibyuvBufferUtils::FlipHorizontally(input, *GetOutput())
                     : LibyuvBufferUtils::FlipVertically(input, *GetOutput());
}

absl::Status FlipOperation::ValidateOutput(const Buffer& input) const {
  if (!input.IsBufferFormatCompatible(*output_)) {
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
  return absl::Status();
}

absl::Status FlipOperation::CreateOutput(const Buffer& input) {
  output_ =
      Buffer::CreateEmpty(input.GetDimension()[0], input.GetDimension()[1],
                          input.GetBufferFormat(), input.GetOrientation());
  return absl::Status();
}

absl::Status ConvertOperation::ProcessImpl(const Buffer& input) {
  return LibyuvBufferUtils::Convert(input, *GetOutput());
}

absl::Status ConvertOperation::ValidateOutput(const Buffer& input) const {
  if (input.GetBufferFormat() == output_->GetBufferFormat()) {
    return absl::InvalidArgumentError("Formats must be different.");
  }

  switch (input.GetBufferFormat()) {
    case BufferFormat::kGrayScale:
      return absl::InvalidArgumentError(
          "Grayscale format does not convert to other formats.");
    case BufferFormat::kRGB:
      if (output_->GetBufferFormat() == BufferFormat::kRGBA) {
        return absl::InvalidArgumentError(
            "RGB format does not convert to RGBA");
      }
      return absl::OkStatus();
    case BufferFormat::kRGBA:
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return absl::OkStatus();
    default:
      return absl::InternalError(absl::StrFormat(
          "Unsupported buffer format: %s.", ToString(input.GetBufferFormat())));
  }
  return absl::Status();
}

absl::Status ConvertOperation::CreateOutput(const Buffer& input) {
  if (!is_format_specified_) {
    return absl::InvalidArgumentError(
        "ConvertOperation: output buffer format is not set.");
  } else {
    output_ =
        Buffer::CreateEmpty(input.GetDimension()[0], input.GetDimension()[1],
                            output_format_, input.GetOrientation());
  }

  return absl::OkStatus();
}

}  // namespace band