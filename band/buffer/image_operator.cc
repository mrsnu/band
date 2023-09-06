#include "band/buffer/image_operator.h"

#include "absl/strings/str_format.h"
#include "band/buffer/libyuv_image_operator.h"
#include "band/buffer/operator.h"
#include "band/logger.h"
#include "image_operator.h"

namespace band {
namespace buffer {

IBufferOperator* Crop::Clone() const { return new Crop(*this); }

IBufferOperator* Resize::Clone() const { return new Resize(*this); }

IBufferOperator* Rotate::Clone() const { return new Rotate(*this); }

Flip::~Flip() {
  if (intermediate_buffer_) {
    delete intermediate_buffer_;
  }
}

IBufferOperator* Flip::Clone() const { return new Flip(*this); }

IBufferOperator* ColorSpaceConvert::Clone() const {
  return new ColorSpaceConvert(*this);
}

AutoConvert::~AutoConvert() {
  // auto convert does not owns output buffer
  // reset output buffer to avoid double free
  output_ = nullptr;
}

IBufferOperator* AutoConvert::Clone() const { return new AutoConvert(*this); }

IBufferOperator::Type Crop::GetOpType() const {
  return IBufferOperator::Type::kImage;
}

IBufferOperator::Type Resize::GetOpType() const {
  return IBufferOperator::Type::kImage;
}

IBufferOperator::Type Rotate::GetOpType() const {
  return IBufferOperator::Type::kImage;
}

IBufferOperator::Type Flip::GetOpType() const {
  return IBufferOperator::Type::kImage;
}

IBufferOperator::Type ColorSpaceConvert::GetOpType() const {
  return IBufferOperator::Type::kImage;
}

IBufferOperator::Type AutoConvert::GetOpType() const { return Type(); }

absl::Status Crop::ProcessImpl(const Buffer& input) {
  BAND_LOG_PROD(
      BAND_LOG_INFO, "Crop: %d x %d (%s, %s) -> %d x %d (%s, %s)",
      input.GetDimension()[0], input.GetDimension()[1],
      ToString(input.GetBufferFormat()), ToString(input.GetDataType()),
      output_->GetDimension()[0], output_->GetDimension()[1],
      ToString(output_->GetBufferFormat()), ToString(output_->GetDataType()));

  return LibyuvImageOperator::Crop(input, x0_, y0_, x1_, y1_, *GetOutput());
}

absl::Status Crop::ValidateInput(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::kRaw) {
    return absl::InvalidArgumentError(
        "Crop: Raw buffer format type is not supported.");
  }

  if (x0_ < 0 || y0_ < 0 || x1_ < 0 || y1_ < 0) {
    return absl::InvalidArgumentError(
        "Crop: negative crop region is not allowed.");
  }

  if (x0_ >= x1_ || y0_ >= y1_) {
    return absl::InvalidArgumentError(
        "Crop: invalid crop region is not allowed.");
  }

  if (x1_ > input.GetDimension()[0] || y1_ > input.GetDimension()[1]) {
    return absl::InvalidArgumentError("Crop: crop region is out of bounds.");
  }

  if (input.GetDataType() != DataType::kUInt8) {
    return absl::InvalidArgumentError(
        "Crop: only UInt8 data type is supported.");
  }

  return absl::OkStatus();
}  // namespace buffer

absl::Status Crop::ValidateOutput(const Buffer& input) const {
  if (!input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Crop: output buffer format type is not "
                        "compatible. %s vs %s",
                        ToString(input.GetBufferFormat()),
                        ToString(output_->GetBufferFormat())));
  }

  if (input.GetDataType() != output_->GetDataType()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Crop: output buffer data type is not "
        "compatible. %s vs %s",
        ToString(input.GetDataType()), ToString(output_->GetDataType())));
  }

  const std::vector<size_t> crop_dimension =
      Buffer::GetCropDimension(x0_, x1_, y0_, y1_);

  if (crop_dimension[0] != output_->GetDimension()[0] ||
      crop_dimension[1] != output_->GetDimension()[1]) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Crop: output buffer dimension is not "
        "compatible. %d x %d vs %d x %d",
        crop_dimension[0], crop_dimension[1], output_->GetDimension()[0],
        output_->GetDimension()[1]));
  }

  return absl::OkStatus();
}

absl::Status Crop::CreateOutput(const Buffer& input) {
  const std::vector<size_t> crop_dimension =
      Buffer::GetCropDimension(x0_, x1_, y0_, y1_);
  output_ = Buffer::CreateEmpty(crop_dimension[0], crop_dimension[1],
                                input.GetBufferFormat(), input.GetDataType(),
                                input.GetOrientation());
  return absl::OkStatus();
}

absl::Status Resize::ProcessImpl(const Buffer& input) {
  BAND_LOG_PROD(BAND_LOG_INFO, "Resize: %d x %d -> %d x %d",
                input.GetDimension()[0], input.GetDimension()[1],
                output_->GetDimension()[0], output_->GetDimension()[1]);
  return LibyuvImageOperator::Resize(input, *GetOutput());
}

absl::Status Resize::ValidateInput(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::kRaw) {
    return absl::InvalidArgumentError(
        "Resize: Raw buffer format type is not supported.");
  }

  if (dims_.size() < 2) {
    return absl::InvalidArgumentError("Resize: invalid dimension size.");
  }

  return absl::OkStatus();
}

absl::Status Resize::ValidateOutput(const Buffer& input) const {
  if (input.GetDimension() == output_->GetDimension()) {
    return absl::InvalidArgumentError(
        "Resize: input and output buffer dimensions "
        "are the same.");
  }

  if (!input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(
        "Resize: input and output buffer formats are not compatible.");
  }

  switch (input.GetBufferFormat()) {
    case BufferFormat::kGrayScale:
    case BufferFormat::kRGB:
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      if (input.GetBufferFormat() != output_->GetBufferFormat()) {
        return absl::InvalidArgumentError(
            "Resize: output buffer format type is not compatible.");
      }
      break;
    case BufferFormat::kRGBA:
      if (output_->GetBufferFormat() != BufferFormat::kRGB &&
          output_->GetBufferFormat() != BufferFormat::kRGBA) {
        return absl::InvalidArgumentError(
            "Resize: output buffer format type is not compatible.");
      }
      break;
    default:
      return absl::InternalError(absl::StrFormat(
          "Unsupported buffer format: %s.", ToString(input.GetBufferFormat())));
  }

  for (size_t i = 0; i < dims_.size(); ++i) {
    if (!IsAuto(i) && dims_[i] != output_->GetDimension()[i]) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Resize: output buffer dimension is not "
                          "compatible. %d != %zd",
                          dims_[i], output_->GetDimension()[i]));
    }
  }

  return absl::Status();
}

absl::Status Resize::CreateOutput(const Buffer& input) {
  if (IsAuto(0) || IsAuto(1)) {
    return absl::InvalidArgumentError(
        "Resize: cannot create output buffer with auto dimension.");
  }

  output_ = Buffer::CreateEmpty(dims_[0], dims_[1], input.GetBufferFormat(),
                                input.GetDataType(), input.GetOrientation());
  return absl::OkStatus();
}

absl::Status Rotate::ProcessImpl(const Buffer& input) {
  BAND_LOG_PROD(BAND_LOG_INFO,
                "Rotate: input dimension: %d x %d, output dimension: %d x "
                "%d, angle: %d",
                input.GetDimension()[0], input.GetDimension()[1],
                output_->GetDimension()[0], output_->GetDimension()[1],
                angle_deg_);

  return LibyuvImageOperator::Rotate(input, angle_deg_, *GetOutput());
}

absl::Status Rotate::ValidateInput(const Buffer& input) const {
  if (input.GetBufferFormat() == BufferFormat::kRaw) {
    return absl::InvalidArgumentError(
        "Rotate: Raw buffer format type is not supported.");
  }
  return absl::OkStatus();
}

absl::Status Rotate::ValidateOutput(const Buffer& input) const {
  if (!input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(
        "Rotate: output buffer format type is not compatible.");
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

absl::Status Rotate::CreateOutput(const Buffer& input) {
  const bool is_dimension_change = (angle_deg_ / 90) % 2 == 1;
  const size_t width =
      is_dimension_change ? input.GetDimension()[1] : input.GetDimension()[0];
  const size_t height =
      is_dimension_change ? input.GetDimension()[0] : input.GetDimension()[1];

  output_ = Buffer::CreateEmpty(width, height, input.GetBufferFormat(),
                                input.GetDataType(), input.GetOrientation());
  return absl::OkStatus();
}

absl::Status Flip::ProcessImpl(const Buffer& input) {
  if (horizontal_ && vertical_) {
    RETURN_IF_ERROR(
        LibyuvImageOperator::FlipHorizontally(input, *intermediate_buffer_));
    return LibyuvImageOperator::FlipVertically(*intermediate_buffer_,
                                               *GetOutput());
  } else if (horizontal_) {
    return LibyuvImageOperator::FlipHorizontally(input, *GetOutput());
  } else {
    return LibyuvImageOperator::FlipVertically(input, *GetOutput());
  }
}

absl::Status Flip::ValidateInput(const Buffer& input) const {
  if (!horizontal_ && !vertical_) {
    return absl::InvalidArgumentError(
        "Flip: either horizontal or vertical flip must be enabled.");
  }

  if (input.GetBufferFormat() == BufferFormat::kRaw) {
    return absl::InvalidArgumentError(
        "Flip: Raw buffer format type is not supported.");
  }

  return absl::Status();
}

absl::Status Flip::ValidateOutput(const Buffer& input) const {
  if (!input.IsBufferFormatCompatible(*output_)) {
    return absl::InvalidArgumentError(
        "Flip: output buffer format type is not compatible.");
  }

  const auto& input_dims = input.GetDimension();
  const auto& output_dims = output_->GetDimension();
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] != output_dims[i]) {
      return absl::InvalidArgumentError(
          "Flip: input and output buffer dimensions must be same.");
    }
  }
  return absl::Status();
}

absl::Status Flip::CreateOutput(const Buffer& input) {
  if (horizontal_ && vertical_) {
    intermediate_buffer_ = Buffer::CreateEmpty(
        input.GetDimension()[0], input.GetDimension()[1],
        input.GetBufferFormat(), input.GetDataType(), input.GetOrientation());
  }

  output_ = Buffer::CreateEmpty(
      input.GetDimension()[0], input.GetDimension()[1], input.GetBufferFormat(),
      input.GetDataType(), input.GetOrientation());
  return absl::Status();
}

absl::Status ColorSpaceConvert::ProcessImpl(const Buffer& input) {
  BAND_LOG_PROD(
      BAND_LOG_INFO, "ColorSpaceConvert: input format: %s, output format: %s",
      ToString(input.GetBufferFormat()), ToString(output_->GetBufferFormat()));

  return LibyuvImageOperator::ColorSpaceConvert(input, *GetOutput());
}

absl::Status ColorSpaceConvert::ValidateOutput(const Buffer& input) const {
  if (input.GetBufferFormat() == output_->GetBufferFormat()) {
    return absl::InvalidArgumentError(
        "ColorSpaceConvert: input and output buffer formats must be "
        "different.");
  }

  if (input.GetDataType() != output_->GetDataType()) {
    return absl::InvalidArgumentError("Data types must be the same.");
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

absl::Status ColorSpaceConvert::CreateOutput(const Buffer& input) {
  if (!is_format_specified_) {
    return absl::InvalidArgumentError(
        "Convert: output buffer format is not set.");
  } else {
    DataType data_type = input.GetDataType();

    if (Buffer::IsYUV(output_format_)) {
      data_type = DataType::kUInt8;
    }

    output_ =
        Buffer::CreateEmpty(input.GetDimension()[0], input.GetDimension()[1],
                            output_format_, data_type, input.GetOrientation());
  }

  return absl::OkStatus();
}

absl::Status AutoConvert::ProcessImpl(const Buffer& input) {
  bool is_processed = false;
  Buffer const* current = &input;
  if (RequiresColorSpaceConvert(input)) {
    RETURN_IF_ERROR(color_space_convert_.Process(input));
    current = color_space_convert_.GetOutput();
    is_processed = true;
  }

  if (RequiresResize(*current)) {
    RETURN_IF_ERROR(resize_.Process(*current));
    current = resize_.GetOutput();
    is_processed = true;
  }

  if (RequiresDataTypeConvert(*current)) {
    RETURN_IF_ERROR(data_type_convert_.Process(*current));
    is_processed = true;
  }

  // if no operation is performed, copy the input to output
  if (!is_processed) {
    RETURN_IF_ERROR(output_->CopyFrom(input));
  }

  return absl::Status();
}

absl::Status AutoConvert::ValidateInput(const Buffer& input) const {
  if (Buffer::IsYUV(input.GetBufferFormat())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input buffer format %d is not supported.",
                        static_cast<int>(input.GetBufferFormat())));
  }

  return absl::Status();
}

absl::Status AutoConvert::ValidateOutput(const Buffer& input) const {
  Buffer const* current = &input;
  if (RequiresColorSpaceConvert(input)) {
    if (!color_space_convert_.GetOutput()) {
      return absl::InvalidArgumentError(
          "color_space_convert_ output is nullptr.");
    }

    RETURN_IF_ERROR(color_space_convert_.ValidateOutput(input));
    current = color_space_convert_.GetOutput();
  }

  if (RequiresResize(*current)) {
    if (!resize_.GetOutput()) {
      return absl::InvalidArgumentError("resize_ output is nullptr.");
    }

    RETURN_IF_ERROR(resize_.ValidateOutput(*current));
    current = resize_.GetOutput();
  }

  if (RequiresDataTypeConvert(*current)) {
    RETURN_IF_ERROR(data_type_convert_.ValidateOutput(*current));
  }

  return absl::Status();
}

absl::Status AutoConvert::CreateOutput(const Buffer& input) {
  IBufferOperator* last_operation = nullptr;

  Buffer const* current = &input;
  if (RequiresColorSpaceConvert(input)) {
    color_space_convert_ = ColorSpaceConvert(output_->GetBufferFormat());
    RETURN_IF_ERROR(color_space_convert_.CreateOutput(*current));
    current = color_space_convert_.GetOutput();
    last_operation = &color_space_convert_;
  }

  if (RequiresResize(*current)) {
    resize_ = Resize(output_->GetDimension()[0], output_->GetDimension()[1]);
    RETURN_IF_ERROR(resize_.CreateOutput(*current));
    current = resize_.GetOutput();
    last_operation = &resize_;
  }

  if (RequiresDataTypeConvert(*current)) {
    data_type_convert_ = DataTypeConvert();
    last_operation = &data_type_convert_;
  }

  if (last_operation) {
    last_operation->SetOutput(output_);
  }

  return absl::OkStatus();
}

void AutoConvert::SetOutput(Buffer* output) {
  // do not set output_assigned_ here, since internal operators will be
  // dependent on the output of its previous operator.
  if (output) {
    output_ = output;
  }
}

bool AutoConvert::RequiresColorSpaceConvert(const Buffer& input) const {
  return input.GetBufferFormat() != output_->GetBufferFormat();
}

bool AutoConvert::RequiresResize(const Buffer& input) const {
  return input.GetDimension() != output_->GetDimension();
}

bool AutoConvert::RequiresDataTypeConvert(const Buffer& input) const {
  return input.GetDataType() != output_->GetDataType();
}

}  // namespace buffer
}  // namespace band