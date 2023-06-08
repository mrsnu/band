/*
Heavily modified from the original source code:
tflite-support/tensorflow_lite_support/cc/task/vision/utils
/libyuv_frame_buffer_utils.cc
by Jingyu Lee <dostos10@gmail.com>
*/

#include "band/tensor/libyuv_operation.h"

#include <stdint.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "libyuv.h"

namespace band {
namespace tensor {

using ::absl::StatusCode;
namespace {

struct YuvData {
  const uint8_t* y_buffer;
  int y_row_stride;
  const uint8_t* u_buffer;
  int uv_row_stride;
  const uint8_t* v_buffer;
};

absl::StatusOr<YuvData> GetYuvDataFromBuffer(const Buffer& buffer) {}

// Converts NV12 `buffer` to the `output_buffer` of the target color space.
// Supported output format includes RGB24 and YV21.
absl::Status ConvertFromNv12(const Buffer& buffer, Buffer* output_buffer) {
  switch (output_buffer->format()) {
    ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                     Buffer::GetYuvDataFromBuffer(*output_buffer));
    case Buffer::Format::kRGB: {
      // The RAW format of Libyuv represents the 8-bit interleaved RGB format in
      // the big endian style with R being the first byte in memory.
      int ret = libyuv::NV12ToRAW(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12ToRAW operation failed.");
      }
      break;
    }
    case Buffer::Format::kRGBA: {
      // The libyuv ABGR format is interleaved RGBA format in memory.
      int ret = libyuv::NV12ToABGR(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12ToABGR operation failed.");
      }
      break;
    }
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::NV12ToI420(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride, const_cast<uint8_t*>(output_data.y_buffer),
          output_data.y_row_stride, const_cast<uint8_t*>(output_data.u_buffer),
          output_data.uv_row_stride, const_cast<uint8_t*>(output_data.v_buffer),
          output_data.uv_row_stride, output_buffer->dimension().width,
          output_buffer->dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12ToI420 operation failed.");
      }
      break;
    }
    case Buffer::Format::kNV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      libyuv::CopyPlane(yuv_data.y_buffer, yuv_data.y_row_stride,
                        const_cast<uint8_t*>(output_data.y_buffer),
                        output_data.y_row_stride, buffer.dimension().width,
                        buffer.dimension().height);
      ASSIGN_OR_RETURN(
          const Buffer::Dimension uv_plane_dimension,
          GetUvPlaneDimension(buffer.dimension(), buffer.format()));
      libyuv::SwapUVPlane(yuv_data.u_buffer, yuv_data.uv_row_stride,
                          const_cast<uint8_t*>(output_data.v_buffer),
                          output_data.uv_row_stride, uv_plane_dimension.width,
                          uv_plane_dimension.height);
      break;
    }
    case Buffer::Format::kGRAY: {
      libyuv::CopyPlane(yuv_data.y_buffer, yuv_data.y_row_stride,
                        const_cast<uint8_t*>(output_buffer->plane(0).buffer),
                        output_buffer->plane(0).stride.row_stride_bytes,
                        output_buffer->dimension().width,
                        output_buffer->dimension().height);
      break;
    }
    default:
      return absl::InternalError(absl::StrFormat("Format %i is not supported.",
                                                 output_buffer->format()));
  }
  return absl::OkStatus();
}

// Converts NV21 `buffer` into the `output_buffer` of the target color space.
// Supported output format includes RGB24 and YV21.
absl::Status ConvertFromNv21(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData yuv_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  switch (output_buffer->format()) {
    case Buffer::Format::kRGB: {
      // The RAW format of Libyuv represents the 8-bit interleaved RGB format in
      // the big endian style with R being the first byte in memory.
      int ret = libyuv::NV21ToRAW(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.v_buffer,
          yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV21ToRAW operation failed.");
      }
      break;
    }
    case Buffer::Format::kRGBA: {
      // The libyuv ABGR format is interleaved RGBA format in memory.
      int ret = libyuv::NV21ToABGR(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.v_buffer,
          yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV21ToABGR operation failed.");
      }
      break;
    }
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::NV21ToI420(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.v_buffer,
          yuv_data.uv_row_stride, const_cast<uint8_t*>(output_data.y_buffer),
          output_data.y_row_stride, const_cast<uint8_t*>(output_data.u_buffer),
          output_data.uv_row_stride, const_cast<uint8_t*>(output_data.v_buffer),
          output_data.uv_row_stride, output_buffer->dimension().width,
          output_buffer->dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV21ToI420 operation failed.");
      }
      break;
    }
    case Buffer::Format::kNV12: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      libyuv::CopyPlane(yuv_data.y_buffer, yuv_data.y_row_stride,
                        const_cast<uint8_t*>(output_data.y_buffer),
                        output_data.y_row_stride, buffer.dimension().width,
                        buffer.dimension().height);
      ASSIGN_OR_RETURN(
          const Buffer::Dimension uv_plane_dimension,
          GetUvPlaneDimension(buffer.dimension(), buffer.format()));
      libyuv::SwapUVPlane(yuv_data.v_buffer, yuv_data.uv_row_stride,
                          const_cast<uint8_t*>(output_data.u_buffer),
                          output_data.uv_row_stride, uv_plane_dimension.width,
                          uv_plane_dimension.height);
      break;
    }
    case Buffer::Format::kGRAY: {
      libyuv::CopyPlane(yuv_data.y_buffer, yuv_data.y_row_stride,
                        const_cast<uint8_t*>(output_buffer->plane(0).buffer),
                        output_buffer->plane(0).stride.row_stride_bytes,
                        output_buffer->dimension().width,
                        output_buffer->dimension().height);
      break;
    }
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.",
                          output_buffer->format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
  return absl::OkStatus();
}

// Converts YV12/YV21 `buffer` to the `output_buffer` of the target color space.
// Supported output format includes RGB24, NV12, and NV21.
absl::Status ConvertFromYv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData yuv_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  switch (output_buffer->format()) {
    case Buffer::Format::kRGB: {
      // The RAW format of Libyuv represents the 8-bit interleaved RGB format in
      // the big endian style with R being the first byte in memory.
      int ret = libyuv::I420ToRAW(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride, yuv_data.v_buffer, yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToRAW operation failed.");
      }
      break;
    }
    case Buffer::Format::kRGBA: {
      // The libyuv ABGR format is interleaved RGBA format in memory.
      int ret = libyuv::I420ToABGR(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride, yuv_data.v_buffer, yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToABGR operation failed.");
      }
      break;
    }
    case Buffer::Format::kNV12: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::I420ToNV12(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride, yuv_data.v_buffer, yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
          output_buffer->dimension().width, output_buffer->dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToNV12 operation failed.");
      }
      break;
    }
    case Buffer::Format::kNV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::I420ToNV21(
          yuv_data.y_buffer, yuv_data.y_row_stride, yuv_data.u_buffer,
          yuv_data.uv_row_stride, yuv_data.v_buffer, yuv_data.uv_row_stride,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
          output_buffer->dimension().width, output_buffer->dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToNV21 operation failed.");
      }
      break;
    }
    case Buffer::Format::kGRAY: {
      libyuv::CopyPlane(yuv_data.y_buffer, yuv_data.y_row_stride,
                        const_cast<uint8_t*>(output_buffer->plane(0).buffer),
                        output_buffer->plane(0).stride.row_stride_bytes,
                        output_buffer->dimension().width,
                        output_buffer->dimension().height);
      break;
    }
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_yuv_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      ASSIGN_OR_RETURN(
          const Buffer::Dimension uv_plane_dimension,
          GetUvPlaneDimension(buffer.dimension(), buffer.format()));
      libyuv::CopyPlane(yuv_data.y_buffer, yuv_data.y_row_stride,
                        const_cast<uint8_t*>(output_yuv_data.y_buffer),
                        output_yuv_data.y_row_stride, buffer.dimension().width,
                        buffer.dimension().height);
      libyuv::CopyPlane(yuv_data.u_buffer, yuv_data.uv_row_stride,
                        const_cast<uint8_t*>(output_yuv_data.u_buffer),
                        output_yuv_data.uv_row_stride, uv_plane_dimension.width,
                        uv_plane_dimension.height);
      libyuv::CopyPlane(yuv_data.v_buffer, yuv_data.uv_row_stride,
                        const_cast<uint8_t*>(output_yuv_data.v_buffer),
                        output_yuv_data.uv_row_stride, uv_plane_dimension.width,
                        uv_plane_dimension.height);
      break;
    }
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.",
                          output_buffer->format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
  return absl::OkStatus();
}

// Resizes YV12/YV21 `buffer` to the target `output_buffer`.
absl::Status ResizeYv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  // TODO(b/151217096): Choose the optimal image resizing filter to optimize
  // the model inference performance.
  int ret = libyuv::I420Scale(
      input_data.y_buffer, input_data.y_row_stride, input_data.u_buffer,
      input_data.uv_row_stride, input_data.v_buffer, input_data.uv_row_stride,
      buffer.dimension().width, buffer.dimension().height,
      const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
      const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
      output_buffer->dimension().width, output_buffer->dimension().height,
      libyuv::FilterMode::kFilterBilinear);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv I420Scale operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Resizes NV12/NV21 `buffer` to the target `output_buffer`.
absl::Status ResizeNv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  const uint8_t* src_uv = input_data.u_buffer;
  const uint8_t* dst_uv = output_data.u_buffer;
  if (buffer.format() == Buffer::Format::kNV21) {
    src_uv = input_data.v_buffer;
    dst_uv = output_data.v_buffer;
  }

  int ret = libyuv::NV12Scale(
      input_data.y_buffer, input_data.y_row_stride, src_uv,
      input_data.uv_row_stride, buffer.dimension().width,
      buffer.dimension().height, const_cast<uint8_t*>(output_data.y_buffer),
      output_data.y_row_stride, const_cast<uint8_t*>(dst_uv),
      output_data.uv_row_stride, output_buffer->dimension().width,
      output_buffer->dimension().height, libyuv::FilterMode::kFilterBilinear);

  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv NV12Scale operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Converts `buffer` to libyuv ARGB format and stores the conversion result
// in `dest_argb`.
absl::Status ConvertRgbToArgb(const Buffer& buffer, uint8_t* dest_argb,
                              int dest_stride_argb) {
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(buffer));
  if (buffer.format() != Buffer::Format::kRGB) {
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "RGB input format is expected.",
                                   TfLiteSupportStatus::kImageProcessingError);
  }

  if (dest_argb == nullptr || dest_stride_argb <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        "Invalid destination arguments for ConvertRgbToArgb.",
        TfLiteSupportStatus::kImageProcessingError);
  }

  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }
  int ret = libyuv::RGB24ToARGB(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      dest_argb, dest_stride_argb, buffer.dimension().width,
      buffer.dimension().height);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv RGB24ToARGB operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Converts `src_argb` in libyuv ARGB format to Buffer::kRGB format and
// stores the conversion result in `output_buffer`.
absl::Status ConvertArgbToRgb(uint8_t* src_argb, int src_stride_argb,
                              Buffer* output_buffer) {
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(*output_buffer));
  if (output_buffer->format() != Buffer::Format::kRGB) {
    return absl::InternalError("RGB input format is expected.");
  }

  if (src_argb == nullptr || src_stride_argb <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInternal, "Invalid source arguments for ConvertArgbToRgb.",
        TfLiteSupportStatus::kImageProcessingError);
  }

  if (output_buffer->plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        output_buffer->format()),
        TfLiteSupportStatus::kImageProcessingError);
  }
  int ret = libyuv::ARGBToRGB24(
      src_argb, src_stride_argb,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes,
      output_buffer->dimension().width, output_buffer->dimension().height);

  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv ARGBToRGB24 operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Converts `buffer` in Buffer::kRGBA format to libyuv ARGB (BGRA in
// memory) format and stores the conversion result in `dest_argb`.
absl::Status ConvertRgbaToArgb(const Buffer& buffer, uint8_t* dest_argb,
                               int dest_stride_argb) {
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(buffer));
  if (buffer.format() != Buffer::Format::kRGBA) {
    return CreateStatusWithPayload(
        StatusCode::kInternal, "RGBA input format is expected.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  if (dest_argb == nullptr || dest_stride_argb <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        "Invalid source arguments for ConvertRgbaToArgb.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

  int ret = libyuv::ABGRToARGB(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      dest_argb, dest_stride_argb, buffer.dimension().width,
      buffer.dimension().height);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kInternal, "Libyuv ABGRToARGB operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Converts kRGB `buffer` to the `output_buffer` of the target color space.
absl::Status ConvertFromRgb(const Buffer& buffer, Buffer* output_buffer) {
  if (output_buffer->format() == Buffer::Format::kGRAY) {
    int ret = libyuv::RAWToJ400(
        buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
        const_cast<uint8_t*>(output_buffer->plane(0).buffer),
        output_buffer->plane(0).stride.row_stride_bytes,
        buffer.dimension().width, buffer.dimension().height);
    if (ret != 0) {
      return CreateStatusWithPayload(
          StatusCode::kInternal, "Libyuv RAWToJ400 operation failed.",
          TfLiteSupportStatus::kImageProcessingBackendError);
    }
    return absl::OkStatus();
  } else if (output_buffer->format() == Buffer::Format::kYV12 ||
             output_buffer->format() == Buffer::Format::kYV21 ||
             output_buffer->format() == Buffer::Format::kNV12 ||
             output_buffer->format() == Buffer::Format::kNV21) {
    // libyuv does not support conversion directly from kRGB to kNV12 / kNV21.
    // For kNV12 / kNV21, the implementation converts the kRGB to I420,
    // then converts I420 to kNV12 / kNV21.
    // TODO(b/153000936): use libyuv::RawToNV12 / libyuv::RawToNV21 when they
    // are ready.
    Buffer::YuvData yuv_data;
    std::unique_ptr<uint8_t[]> tmp_yuv_buffer;
    std::unique_ptr<Buffer> yuv_frame_buffer;
    if (output_buffer->format() == Buffer::Format::kNV12 ||
        output_buffer->format() == Buffer::Format::kNV21) {
      tmp_yuv_buffer = absl::make_unique<uint8_t[]>(
          GetBufferByteSize(buffer.dimension(), output_buffer->format()));
      ASSIGN_OR_RETURN(
          yuv_frame_buffer,
          CreateFromRawBuffer(tmp_yuv_buffer.get(), buffer.dimension(),
                              Buffer::Format::kYV21,
                              output_buffer->orientation()));
      ASSIGN_OR_RETURN(yuv_data,
                       Buffer::GetYuvDataFromBuffer(*yuv_frame_buffer));
    } else {
      ASSIGN_OR_RETURN(yuv_data, Buffer::GetYuvDataFromBuffer(*output_buffer));
    }
    int ret = libyuv::RAWToI420(
        buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
        const_cast<uint8_t*>(yuv_data.y_buffer), yuv_data.y_row_stride,
        const_cast<uint8_t*>(yuv_data.u_buffer), yuv_data.uv_row_stride,
        const_cast<uint8_t*>(yuv_data.v_buffer), yuv_data.uv_row_stride,
        buffer.dimension().width, buffer.dimension().height);
    if (ret != 0) {
      return CreateStatusWithPayload(
          StatusCode::kInternal, "Libyuv RAWToI420 operation failed.",
          TfLiteSupportStatus::kImageProcessingBackendError);
    }
    if (output_buffer->format() == Buffer::Format::kNV12 ||
        output_buffer->format() == Buffer::Format::kNV21) {
      return ConvertFromYv(*yuv_frame_buffer, output_buffer);
    }
    return absl::OkStatus();
  }
  return CreateStatusWithPayload(
      StatusCode::kInternal,
      absl::StrFormat("Format %i is not supported.", output_buffer->format()),
      TfLiteSupportStatus::kImageProcessingError);
}

// Converts kRGBA `buffer` to the `output_buffer` of the target color space.
absl::Status ConvertFromRgba(const Buffer& buffer, Buffer* output_buffer) {
  switch (output_buffer->format()) {
    case Buffer::Format::kGRAY: {
      // libyuv does not support convert kRGBA (ABGR) foramat. In this method,
      // the implementation converts kRGBA format to ARGB and use ARGB buffer
      // for conversion.
      // TODO(b/141181395): Use libyuv::ABGRToJ400 when it is ready.

      // Convert kRGBA to ARGB
      int argb_buffer_size =
          GetBufferByteSize(buffer.dimension(), Buffer::Format::kRGBA);
      auto argb_buffer = absl::make_unique<uint8_t[]>(argb_buffer_size);
      const int argb_row_bytes = buffer.dimension().width * kRgbaPixelBytes;
      RETURN_IF_ERROR(
          ConvertRgbaToArgb(buffer, argb_buffer.get(), argb_row_bytes));

      // Convert ARGB to kGRAY
      int ret = libyuv::ARGBToJ400(
          argb_buffer.get(), argb_row_bytes,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ARGBToJ400 operation failed.");
      }
      break;
    }
    case Buffer::Format::kNV12: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::ABGRToNV12(
          buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToNV12 operation failed.");
      }
      break;
    }
    case Buffer::Format::kNV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::ABGRToNV21(
          buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToNV21 operation failed.");
      }
      break;
    }
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21: {
      ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                       Buffer::GetYuvDataFromBuffer(*output_buffer));
      int ret = libyuv::ABGRToI420(
          buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
          const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToI420 operation failed.");
      }
      break;
    }
    case Buffer::Format::kRGB: {
      // ARGB is BGRA in memory and RGB24 is BGR in memory. The removal of the
      // alpha channel will not impact the RGB ordering.
      int ret = libyuv::ARGBToRGB24(
          buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
          const_cast<uint8_t*>(output_buffer->plane(0).buffer),
          output_buffer->plane(0).stride.row_stride_bytes,
          buffer.dimension().width, buffer.dimension().height);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToRGB24 operation failed.");
      }
      break;
    }
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Convert Rgba to format %i is not supported.",
                          output_buffer->format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
  return absl::OkStatus();
}

// Returns libyuv rotation based on counter-clockwise angle_deg.
libyuv::RotationMode GetLibyuvRotationMode(int angle_deg) {
  switch (angle_deg) {
    case 90:
      return libyuv::kRotate270;
    case 270:
      return libyuv::kRotate90;
    case 180:
      return libyuv::kRotate180;
    default:
      return libyuv::kRotate0;
  }
}

absl::Status RotateRgba(const Buffer& buffer, int angle_deg,
                        Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

  // libyuv::ARGBRotate assumes RGBA buffer is in the interleaved format.
  int ret = libyuv::ARGBRotate(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes, buffer.dimension().width,
      buffer.dimension().height, GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv ARGBRotate operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

absl::Status RotateRgb(const Buffer& buffer, int angle_deg,
                       Buffer* output_buffer) {
  // libyuv does not support rotate kRGB (RGB24) foramat. In this method, the
  // implementation converts kRGB format to ARGB and use ARGB buffer for
  // rotation. The result is then convert back to RGB.

  // Convert RGB to ARGB
  int argb_buffer_size =
      GetBufferByteSize(buffer.dimension(), Buffer::Format::kRGBA);
  auto argb_buffer = absl::make_unique<uint8_t[]>(argb_buffer_size);
  const int argb_row_bytes = buffer.dimension().width * kRgbaPixelBytes;
  RETURN_IF_ERROR(ConvertRgbToArgb(buffer, argb_buffer.get(), argb_row_bytes));

  // Rotate ARGB
  auto argb_rotated_buffer = absl::make_unique<uint8_t[]>(argb_buffer_size);
  int rotated_row_bytes = output_buffer->dimension().width * kRgbaPixelBytes;
  // TODO(b/151954340): Optimize the current implementation by utilizing
  // ARGBMirror for 180 degree rotation.
  int ret = libyuv::ARGBRotate(
      argb_buffer.get(), argb_row_bytes, argb_rotated_buffer.get(),
      rotated_row_bytes, buffer.dimension().width, buffer.dimension().height,
      GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv ARGBRotate operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  // Convert ARGB to RGB
  return ConvertArgbToRgb(argb_rotated_buffer.get(), rotated_row_bytes,
                          output_buffer);
}

absl::Status RotateGray(const Buffer& buffer, int angle_deg,
                        Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }
  int ret = libyuv::RotatePlane(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes, buffer.dimension().width,
      buffer.dimension().height, GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv RotatePlane operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Rotates YV12/YV21 frame buffer.
absl::Status RotateYv(const Buffer& buffer, int angle_deg,
                      Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  int ret = libyuv::I420Rotate(
      input_data.y_buffer, input_data.y_row_stride, input_data.u_buffer,
      input_data.uv_row_stride, input_data.v_buffer, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
      const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
      buffer.dimension().width, buffer.dimension().height,
      GetLibyuvRotationMode(angle_deg));
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv I420Rotate operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Rotates NV12/NV21 frame buffer.
// TODO(b/152097364): Refactor NV12/NV21 rotation after libyuv explicitly
// support that.
absl::Status RotateNv(const Buffer& buffer, int angle_deg,
                      Buffer* output_buffer) {
  if (buffer.format() != Buffer::Format::kNV12 &&
      buffer.format() != Buffer::Format::kNV21) {
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "kNV12 or kNV21 input formats are expected.",
                                   TfLiteSupportStatus::kImageProcessingError);
  }
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  const int rotated_buffer_size =
      GetBufferByteSize(output_buffer->dimension(), Buffer::Format::kYV21);
  auto rotated_yuv_raw_buffer =
      absl::make_unique<uint8_t[]>(rotated_buffer_size);
  ASSIGN_OR_RETURN(std::unique_ptr<Buffer> rotated_yuv_buffer,
                   CreateFromRawBuffer(rotated_yuv_raw_buffer.get(),
                                       output_buffer->dimension(),
                                       /*target_format=*/Buffer::Format::kYV21,
                                       output_buffer->orientation()));
  ASSIGN_OR_RETURN(Buffer::YuvData rotated_yuv_data,
                   Buffer::GetYuvDataFromBuffer(*rotated_yuv_buffer));
  // Get the first chroma plane and use it as the u plane. This is a workaround
  // for optimizing NV21 rotation. For NV12, the implementation is logical
  // correct. For NV21, use v plane as u plane will make the UV planes swapped
  // in the intermediate rotated I420 frame. The output buffer is finally built
  // by merging the swapped UV planes which produces V first interleaved UV
  // buffer.
  const uint8_t* chroma_buffer = buffer.format() == Buffer::Format::kNV12
                                     ? input_data.u_buffer
                                     : input_data.v_buffer;
  // Rotate the Y plane and store into the Y plane in `output_buffer`. Rotate
  // the interleaved UV plane and store into the interleaved UV plane in
  // `rotated_yuv_buffer`.
  int ret = libyuv::NV12ToI420Rotate(
      input_data.y_buffer, input_data.y_row_stride, chroma_buffer,
      input_data.uv_row_stride, const_cast<uint8_t*>(output_data.y_buffer),
      output_data.y_row_stride, const_cast<uint8_t*>(rotated_yuv_data.u_buffer),
      rotated_yuv_data.uv_row_stride,
      const_cast<uint8_t*>(rotated_yuv_data.v_buffer),
      rotated_yuv_data.uv_row_stride, buffer.dimension().width,
      buffer.dimension().height, GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv Nv12ToI420Rotate operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  // Merge rotated UV planes into the output buffer. For NV21, the UV buffer of
  // the intermediate I420 frame is swapped. MergeUVPlane builds the interleaved
  // VU buffer for NV21 by putting the U plane in the I420 frame which is
  // actually the V plane from the input buffer first.
  const uint8_t* output_chroma_buffer = buffer.format() == Buffer::Format::kNV12
                                            ? output_data.u_buffer
                                            : output_data.v_buffer;
  // The width and height arguments of `libyuv::MergeUVPlane()` represent the
  // width and height of the UV planes.
  libyuv::MergeUVPlane(
      rotated_yuv_data.u_buffer, rotated_yuv_data.uv_row_stride,
      rotated_yuv_data.v_buffer, rotated_yuv_data.uv_row_stride,
      const_cast<uint8_t*>(output_chroma_buffer), output_data.uv_row_stride,
      (output_buffer->dimension().width + 1) / 2,
      (output_buffer->dimension().height + 1) / 2);
  return absl::OkStatus();
}

// This method only supports kGRAY, kRGB, and kRGBA format.
absl::Status FlipPlaneVertically(const Buffer& buffer, Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

  ASSIGN_OR_RETURN(int pixel_stride, GetPixelStrides(buffer.format()));

  // Flip vertically is achieved by passing in negative height.
  libyuv::CopyPlane(buffer.plane(0).buffer,
                    buffer.plane(0).stride.row_stride_bytes,
                    const_cast<uint8_t*>(output_buffer->plane(0).buffer),
                    output_buffer->plane(0).stride.row_stride_bytes,
                    output_buffer->dimension().width * pixel_stride,
                    -output_buffer->dimension().height);

  return absl::OkStatus();
}

// This method only supports kGRAY, kRGBA, and kRGB formats.
absl::Status CropPlane(const Buffer& buffer, int x0, int y0, int x1, int y1,
                       Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

  ASSIGN_OR_RETURN(int pixel_stride, GetPixelStrides(buffer.format()));
  Buffer::Dimension crop_dimension = GetCropDimension(x0, x1, y0, y1);

  // Cropping is achieved by adjusting origin to (x0, y0).
  int adjusted_offset =
      buffer.plane(0).stride.row_stride_bytes * y0 + x0 * pixel_stride;

  libyuv::CopyPlane(buffer.plane(0).buffer + adjusted_offset,
                    buffer.plane(0).stride.row_stride_bytes,
                    const_cast<uint8_t*>(output_buffer->plane(0).buffer),
                    output_buffer->plane(0).stride.row_stride_bytes,
                    crop_dimension.width * pixel_stride, crop_dimension.height);

  return absl::OkStatus();
}

// Crops NV12/NV21 Buffer to the subregion defined by the top left pixel
// position (x0, y0) and the bottom right pixel position (x1, y1).
absl::Status CropNv(const Buffer& buffer, int x0, int y0, int x1, int y1,
                    Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  // Crop Y plane by copying the buffer with the origin offset to (x0, y0).
  int crop_offset_y = input_data.y_row_stride * y0 + x0;
  int crop_width = x1 - x0 + 1;
  int crop_height = y1 - y0 + 1;
  libyuv::CopyPlane(input_data.y_buffer + crop_offset_y,
                    input_data.y_row_stride,
                    const_cast<uint8_t*>(output_data.y_buffer),
                    output_data.y_row_stride, crop_width, crop_height);
  // Crop chroma plane by copying the buffer with the origin offset to
  // (x0 / 2, y0 / 2);
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  int crop_offset_chroma = input_data.uv_row_stride * (y0 / 2) +
                           input_data.uv_pixel_stride * (x0 / 2);
  ASSIGN_OR_RETURN(const uint8_t* input_chroma_buffer, GetUvRawBuffer(buffer));
  ASSIGN_OR_RETURN(const uint8_t* output_chroma_buffer,
                   GetUvRawBuffer(*output_buffer));
  libyuv::CopyPlane(
      input_chroma_buffer + crop_offset_chroma, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_chroma_buffer), output_data.uv_row_stride,
      /*width=*/(crop_width + 1) / 2 * 2, /*height=*/(crop_height + 1) / 2);
  return absl::OkStatus();
}

// Crops YV12/YV21 Buffer to the subregion defined by the top left pixel
// position (x0, y0) and the bottom right pixel position (x1, y1).
absl::Status CropYv(const Buffer& buffer, int x0, int y0, int x1, int y1,
                    Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  // Crop Y plane by copying the buffer with the origin offset to (x0, y0).
  int crop_offset_y = input_data.y_row_stride * y0 + x0;
  Buffer::Dimension crop_dimension = GetCropDimension(x0, x1, y0, y1);
  libyuv::CopyPlane(
      input_data.y_buffer + crop_offset_y, input_data.y_row_stride,
      const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
      crop_dimension.width, crop_dimension.height);
  // Crop U plane by copying the buffer with the origin offset to
  // (x0 / 2, y0 / 2).
  ASSIGN_OR_RETURN(const Buffer::Dimension crop_uv_dimension,
                   GetUvPlaneDimension(crop_dimension, buffer.format()));
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  int crop_offset_chroma = input_data.uv_row_stride * (y0 / 2) +
                           input_data.uv_pixel_stride * (x0 / 2);
  libyuv::CopyPlane(
      input_data.u_buffer + crop_offset_chroma, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
      crop_uv_dimension.width, crop_uv_dimension.height);
  // Crop V plane by copying the buffer with the origin offset to
  // (x0 / 2, y0 / 2);
  libyuv::CopyPlane(
      input_data.v_buffer + crop_offset_chroma, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
      /*width=*/(crop_dimension.width + 1) / 2,
      /*height=*/(crop_dimension.height + 1) / 2);
  return absl::OkStatus();
}

absl::Status CropResizeYuv(const Buffer& buffer, int x0, int y0, int x1, int y1,
                           Buffer* output_buffer) {
  Buffer::Dimension crop_dimension = GetCropDimension(x0, x1, y0, y1);
  if (crop_dimension == output_buffer->dimension()) {
    switch (buffer.format()) {
      case Buffer::Format::kNV12:
      case Buffer::Format::kNV21:
        return CropNv(buffer, x0, y0, x1, y1, output_buffer);
      case Buffer::Format::kYV12:
      case Buffer::Format::kYV21:
        return CropYv(buffer, x0, y0, x1, y1, output_buffer);
      default:
        return CreateStatusWithPayload(
            StatusCode::kInternal,
            absl::StrFormat("Format %i is not supported.", buffer.format()),
            TfLiteSupportStatus::kImageProcessingError);
    }
  }
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  // Cropping YUV planes by offsetting the origins of each plane.
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  const int plane_y_offset = input_data.y_row_stride * y0 + x0;
  const int plane_uv_offset = input_data.uv_row_stride * (y0 / 2) +
                              input_data.uv_pixel_stride * (x0 / 2);
  Buffer::Plane cropped_plane_y = {
      /*buffer=*/input_data.y_buffer + plane_y_offset,
      /*stride=*/{input_data.y_row_stride, /*pixel_stride_bytes=*/1}};
  Buffer::Plane cropped_plane_u = {
      /*buffer=*/input_data.u_buffer + plane_uv_offset,
      /*stride=*/{input_data.uv_row_stride, input_data.uv_pixel_stride}};
  Buffer::Plane cropped_plane_v = {
      /*buffer=*/input_data.v_buffer + plane_uv_offset,
      /*stride=*/{input_data.uv_row_stride, input_data.uv_pixel_stride}};

  switch (buffer.format()) {
    case Buffer::Format::kNV12: {
      std::unique_ptr<Buffer> cropped_buffer =
          Buffer::Create({cropped_plane_y, cropped_plane_u, cropped_plane_v},
                         crop_dimension, buffer.format(), buffer.orientation());
      return ResizeNv(*cropped_buffer, output_buffer);
    }
    case Buffer::Format::kNV21: {
      std::unique_ptr<Buffer> cropped_buffer =
          Buffer::Create({cropped_plane_y, cropped_plane_v, cropped_plane_u},
                         crop_dimension, buffer.format(), buffer.orientation());
      return ResizeNv(*cropped_buffer, output_buffer);
    }
    case Buffer::Format::kYV12: {
      std::unique_ptr<Buffer> cropped_buffer =
          Buffer::Create({cropped_plane_y, cropped_plane_v, cropped_plane_u},
                         crop_dimension, buffer.format(), buffer.orientation());
      return ResizeYv(*cropped_buffer, output_buffer);
    }
    case Buffer::Format::kYV21: {
      std::unique_ptr<Buffer> cropped_buffer =
          Buffer::Create({cropped_plane_y, cropped_plane_u, cropped_plane_v},
                         crop_dimension, buffer.format(), buffer.orientation());
      return ResizeYv(*cropped_buffer, output_buffer);
    }
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
  return absl::OkStatus();
}

absl::Status FlipHorizontallyRgba(const Buffer& buffer, Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

  int ret = libyuv::ARGBMirror(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes,
      output_buffer->dimension().width, output_buffer->dimension().height);

  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv ARGBMirror operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  return absl::OkStatus();
}

// Flips `buffer` horizontally and store the result in `output_buffer`. This
// method assumes all buffers have pixel stride equals to 1.
absl::Status FlipHorizontallyPlane(const Buffer& buffer,
                                   Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }
  libyuv::MirrorPlane(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes,
      output_buffer->dimension().width, output_buffer->dimension().height);

  return absl::OkStatus();
}

absl::Status ResizeRgb(const Buffer& buffer, Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

  // libyuv doesn't support scale kRGB (RGB24) foramat. In this method,
  // the implementation converts kRGB format to ARGB and use ARGB buffer for
  // scaling. The result is then convert back to RGB.

  // Convert RGB to ARGB
  int argb_buffer_size =
      GetBufferByteSize(buffer.dimension(), Buffer::Format::kRGBA);
  auto argb_buffer = absl::make_unique<uint8_t[]>(argb_buffer_size);
  const int argb_row_bytes = buffer.dimension().width * kRgbaPixelBytes;
  RETURN_IF_ERROR(ConvertRgbToArgb(buffer, argb_buffer.get(), argb_row_bytes));

  // Resize ARGB
  int resized_argb_buffer_size =
      GetBufferByteSize(output_buffer->dimension(), Buffer::Format::kRGBA);
  auto resized_argb_buffer =
      absl::make_unique<uint8_t[]>(resized_argb_buffer_size);
  int resized_argb_row_bytes =
      output_buffer->dimension().width * kRgbaPixelBytes;
  int ret = libyuv::ARGBScale(
      argb_buffer.get(), argb_row_bytes, buffer.dimension().width,
      buffer.dimension().height, resized_argb_buffer.get(),
      resized_argb_row_bytes, output_buffer->dimension().width,
      output_buffer->dimension().height, libyuv::FilterMode::kFilterBilinear);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv ARGBScale operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  // Convert ARGB to RGB
  return ConvertArgbToRgb(resized_argb_buffer.get(), resized_argb_row_bytes,
                          output_buffer);
}

// Horizontally flip `buffer` and store the result in `output_buffer`.
absl::Status FlipHorizontallyRgb(const Buffer& buffer, Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }

#if LIBYUV_VERSION >= 1747
  int ret = libyuv::RGB24Mirror(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes, buffer.dimension().width,
      buffer.dimension().height);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv RGB24Mirror operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  return absl::OkStatus();
#else
#error LibyuvBufferUtils requires LIBYUV_VERSION 1747 or above
#endif  // LIBYUV_VERSION >= 1747
}

absl::Status ResizeRgba(const Buffer& buffer, Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }
  int ret = libyuv::ARGBScale(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      buffer.dimension().width, buffer.dimension().height,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes,
      output_buffer->dimension().width, output_buffer->dimension().height,
      libyuv::FilterMode::kFilterBilinear);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv ARGBScale operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Flips NV12/NV21 Buffer horizontally.
absl::Status FlipHorizontallyNv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  ASSIGN_OR_RETURN(const uint8_t* input_chroma_buffer, GetUvRawBuffer(buffer));
  ASSIGN_OR_RETURN(const uint8_t* output_chroma_buffer,
                   GetUvRawBuffer(*output_buffer));

  int ret = libyuv::NV12Mirror(
      input_data.y_buffer, input_data.y_row_stride, input_chroma_buffer,
      input_data.uv_row_stride, const_cast<uint8_t*>(output_data.y_buffer),
      output_data.y_row_stride, const_cast<uint8_t*>(output_chroma_buffer),
      output_data.uv_row_stride, buffer.dimension().width,
      buffer.dimension().height);

  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv NV12Mirror operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  return absl::OkStatus();
}

// Flips YV12/YV21 Buffer horizontally.
absl::Status FlipHorizontallyYv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  int ret = libyuv::I420Mirror(
      input_data.y_buffer, input_data.y_row_stride, input_data.u_buffer,
      input_data.uv_row_stride, input_data.v_buffer, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
      const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
      buffer.dimension().width, buffer.dimension().height);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv I420Mirror operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }

  return absl::OkStatus();
}

// Flips NV12/NV21 Buffer vertically.
absl::Status FlipVerticallyNv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  // Flip Y plane vertically by passing a negative height.
  libyuv::CopyPlane(input_data.y_buffer, input_data.y_row_stride,
                    const_cast<uint8_t*>(output_data.y_buffer),
                    output_data.y_row_stride, buffer.dimension().width,
                    -output_buffer->dimension().height);
  // Flip UV plane vertically by passing a negative height.
  ASSIGN_OR_RETURN(const uint8_t* input_chroma_buffer, GetUvRawBuffer(buffer));
  ASSIGN_OR_RETURN(const uint8_t* output_chroma_buffer,
                   GetUvRawBuffer(*output_buffer));
  ASSIGN_OR_RETURN(const Buffer::Dimension uv_plane_dimension,
                   GetUvPlaneDimension(buffer.dimension(), buffer.format()));
  libyuv::CopyPlane(
      input_chroma_buffer, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_chroma_buffer), output_data.uv_row_stride,
      /*width=*/uv_plane_dimension.width * 2, -uv_plane_dimension.height);
  return absl::OkStatus();
}

// Flips NV12/NV21 Buffer vertically.
absl::Status FlipVerticallyYv(const Buffer& buffer, Buffer* output_buffer) {
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));
  // Flip buffer vertically by passing a negative height.
  int ret = libyuv::I420Copy(
      input_data.y_buffer, input_data.y_row_stride, input_data.u_buffer,
      input_data.uv_row_stride, input_data.v_buffer, input_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
      const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
      const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
      buffer.dimension().width, -buffer.dimension().height);
  if (ret != 0) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Libyuv I420Copy operation failed.",
        TfLiteSupportStatus::kImageProcessingBackendError);
  }
  return absl::OkStatus();
}

// Resize `buffer` to metadata defined in `output_buffer`. This
// method assumes buffer has pixel stride equals to 1 (grayscale equivalent).
absl::Status ResizeGray(const Buffer& buffer, Buffer* output_buffer) {
  if (buffer.plane_count() > 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.format()),
        TfLiteSupportStatus::kImageProcessingError);
  }
  libyuv::ScalePlane(
      buffer.plane(0).buffer, buffer.plane(0).stride.row_stride_bytes,
      buffer.dimension().width, buffer.dimension().height,
      const_cast<uint8_t*>(output_buffer->plane(0).buffer),
      output_buffer->plane(0).stride.row_stride_bytes,
      output_buffer->dimension().width, output_buffer->dimension().height,
      libyuv::FilterMode::kFilterBilinear);
  return absl::OkStatus();
}

// This method only supports kGRAY, kRGBA, and kRGB formats.
absl::Status CropResize(const Buffer& buffer, int x0, int y0, int x1, int y1,
                        Buffer* output_buffer) {
  Buffer::Dimension crop_dimension = GetCropDimension(x0, x1, y0, y1);
  if (crop_dimension == output_buffer->dimension()) {
    return CropPlane(buffer, x0, y0, x1, y1, output_buffer);
  }

  ASSIGN_OR_RETURN(int pixel_stride, GetPixelStrides(buffer.format()));
  // Cropping is achieved by adjusting origin to (x0, y0).
  int adjusted_offset =
      buffer.plane(0).stride.row_stride_bytes * y0 + x0 * pixel_stride;
  Buffer::Plane plane = {
      /*buffer=*/buffer.plane(0).buffer + adjusted_offset,
      /*stride=*/{buffer.plane(0).stride.row_stride_bytes, pixel_stride}};
  auto adjusted_buffer =
      Buffer::Create({plane}, crop_dimension, buffer.format(),
                     buffer.orientation(), buffer.timestamp());

  switch (buffer.format()) {
    case Buffer::Format::kRGB:
      return ResizeRgb(*adjusted_buffer, output_buffer);
    case Buffer::Format::kRGBA:
      return ResizeRgba(*adjusted_buffer, output_buffer);
    case Buffer::Format::kGRAY:
      return ResizeGray(*adjusted_buffer, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

// Returns the scaled dimension of the input_size best fit within the
// output_size bound while respecting the aspect ratio.
Buffer::Dimension GetScaledDimension(Buffer::Dimension input_size,
                                     Buffer::Dimension output_size) {
  int original_width = input_size.width;
  int original_height = input_size.height;
  int bound_width = output_size.width;
  int bound_height = output_size.height;
  int new_width = original_width;
  int new_height = original_height;

  // Try to fit the width first.
  new_width = bound_width;
  new_height = (new_width * original_height) / original_width;

  // Try to fit the height if needed.
  if (new_height > bound_height) {
    new_height = bound_height;
    new_width = (new_height * original_width) / original_height;
  }
  return Buffer::Dimension{.width = new_width, .height = new_height};
}

// This method only supports kGRAY, kRGBA, and kRGB formats.
absl::Status UniformCropResizePlane(const Buffer& buffer,
                                    std::vector<int> crop_coordinates,
                                    Buffer* output_buffer) {
  int x0 = 0, y0 = 0;
  Buffer::Dimension input_dimension = buffer.dimension();
  if (!crop_coordinates.empty()) {
    x0 = crop_coordinates[0];
    y0 = crop_coordinates[1];
    input_dimension =
        GetCropDimension(x0, crop_coordinates[2], y0, crop_coordinates[3]);
  }
  if (input_dimension == output_buffer->dimension()) {
    // Cropping only case.
    return CropPlane(buffer, x0, y0, crop_coordinates[2], crop_coordinates[3],
                     output_buffer);
  }

  // Cropping is achieved by adjusting origin to (x0, y0).
  ASSIGN_OR_RETURN(int pixel_stride, GetPixelStrides(buffer.format()));
  int adjusted_offset =
      buffer.plane(0).stride.row_stride_bytes * y0 + x0 * pixel_stride;
  Buffer::Plane plane = {
      /*buffer=*/buffer.plane(0).buffer + adjusted_offset,
      /*stride=*/{buffer.plane(0).stride.row_stride_bytes, pixel_stride}};
  auto adjusted_buffer =
      Buffer::Create({plane}, input_dimension, buffer.format(),
                     buffer.orientation(), buffer.timestamp());

  // Uniform resize is achieved by adjusting the resize dimension to fit the
  // output_buffer and respect the input aspect ratio at the same time. We
  // create an intermediate output buffer with adjusted dimension and point its
  // backing buffer to the output_buffer. Note the stride information on the
  // adjusted_output_buffer is not used in the Resize* methods.
  Buffer::Dimension adjusted_dimension =
      GetScaledDimension(input_dimension, output_buffer->dimension());
  Buffer::Plane output_plane = {/*buffer=*/output_buffer->plane(0).buffer,
                                /*stride=*/output_buffer->plane(0).stride};
  auto adjusted_output_buffer = Buffer::Create(
      {output_plane}, adjusted_dimension, output_buffer->format(),
      output_buffer->orientation(), output_buffer->timestamp());

  switch (buffer.format()) {
    case Buffer::Format::kRGB:
      return ResizeRgb(*adjusted_buffer, adjusted_output_buffer.get());
    case Buffer::Format::kRGBA:
      return ResizeRgba(*adjusted_buffer, adjusted_output_buffer.get());
    case Buffer::Format::kGRAY:
      return ResizeGray(*adjusted_buffer, adjusted_output_buffer.get());
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

absl::Status UniformCropResizeYuv(const Buffer& buffer,
                                  std::vector<int> crop_coordinates,
                                  Buffer* output_buffer) {
  int x0 = 0, y0 = 0;
  Buffer::Dimension input_dimension = buffer.dimension();
  if (!crop_coordinates.empty()) {
    x0 = crop_coordinates[0];
    y0 = crop_coordinates[1];
    input_dimension =
        GetCropDimension(x0, crop_coordinates[2], y0, crop_coordinates[3]);
  }
  if (input_dimension == output_buffer->dimension()) {
    // Cropping only case.
    int x1 = crop_coordinates[2];
    int y1 = crop_coordinates[3];
    switch (buffer.format()) {
      case Buffer::Format::kNV12:
      case Buffer::Format::kNV21:
        return CropNv(buffer, x0, y0, x1, y1, output_buffer);
      case Buffer::Format::kYV12:
      case Buffer::Format::kYV21:
        return CropYv(buffer, x0, y0, x1, y1, output_buffer);
      default:
        return CreateStatusWithPayload(
            StatusCode::kInternal,
            absl::StrFormat("Format %i is not supported.", buffer.format()),
            TfLiteSupportStatus::kImageProcessingError);
    }
  }

  // Cropping is achieved by adjusting origin to (x0, y0).
  ASSIGN_OR_RETURN(Buffer::YuvData input_data,
                   Buffer::GetYuvDataFromBuffer(buffer));
  // Cropping YUV planes by offsetting the origins of each plane.
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  const int plane_y_offset = input_data.y_row_stride * y0 + x0;
  const int plane_uv_offset = input_data.uv_row_stride * (y0 / 2) +
                              input_data.uv_pixel_stride * (x0 / 2);
  Buffer::Plane adjusted_plane_y = {
      /*buffer=*/input_data.y_buffer + plane_y_offset,
      /*stride=*/{input_data.y_row_stride, /*pixel_stride_bytes=*/1}};
  Buffer::Plane adjusted_plane_u = {
      /*buffer=*/input_data.u_buffer + plane_uv_offset,
      /*stride=*/{input_data.uv_row_stride, input_data.uv_pixel_stride}};
  Buffer::Plane adjusted_plane_v = {
      /*buffer=*/input_data.v_buffer + plane_uv_offset,
      /*stride=*/{input_data.uv_row_stride, input_data.uv_pixel_stride}};

  // Uniform resize is achieved by adjusting the resize dimension to fit the
  // output_buffer and respect the input aspect ratio at the same time. For
  // YUV formats, we need access to the actual output dimension to get the
  // correct address of each plane. For this, we are not calling ResizeNv or
  // ResizeYv but the libyuv scale methods directly.
  Buffer::Dimension adjusted_dimension =
      GetScaledDimension(input_dimension, output_buffer->dimension());
  ASSIGN_OR_RETURN(Buffer::YuvData output_data,
                   Buffer::GetYuvDataFromBuffer(*output_buffer));

  switch (buffer.format()) {
    case Buffer::Format::kNV12: {
      int ret = libyuv::NV12Scale(
          adjusted_plane_y.buffer, adjusted_plane_y.stride.row_stride_bytes,
          adjusted_plane_u.buffer, adjusted_plane_u.stride.row_stride_bytes,
          input_dimension.width, input_dimension.height,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
          adjusted_dimension.width, adjusted_dimension.height,
          libyuv::FilterMode::kFilterBilinear);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12Scale operation failed.");
      }
      return absl::OkStatus();
    }
    case Buffer::Format::kNV21: {
      int ret = libyuv::NV12Scale(
          adjusted_plane_y.buffer, adjusted_plane_y.stride.row_stride_bytes,
          adjusted_plane_v.buffer, adjusted_plane_v.stride.row_stride_bytes,
          input_dimension.width, input_dimension.height,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
          adjusted_dimension.width, adjusted_dimension.height,
          libyuv::FilterMode::kFilterBilinear);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12Scale operation failed.");
      }
      return absl::OkStatus();
    }
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21: {
      int ret = libyuv::I420Scale(
          adjusted_plane_y.buffer, adjusted_plane_y.stride.row_stride_bytes,
          adjusted_plane_u.buffer, adjusted_plane_u.stride.row_stride_bytes,
          adjusted_plane_v.buffer, adjusted_plane_v.stride.row_stride_bytes,
          input_dimension.width, input_dimension.height,
          const_cast<uint8_t*>(output_data.y_buffer), output_data.y_row_stride,
          const_cast<uint8_t*>(output_data.u_buffer), output_data.uv_row_stride,
          const_cast<uint8_t*>(output_data.v_buffer), output_data.uv_row_stride,
          adjusted_dimension.width, adjusted_dimension.height,
          libyuv::FilterMode::kFilterBilinear);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420Scale operation failed.");
      }
      return absl::OkStatus();
    }
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status LibyuvBufferUtils::Crop(const Buffer& buffer, int x0, int y0,
                                     int x1, int y1, Buffer* output_buffer) {
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(buffer));
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(*output_buffer));
  RETURN_IF_ERROR(
      ValidateCropBufferInputs(buffer, *output_buffer, x0, y0, x1, y1));
  RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case Buffer::Format::kRGBA:
    case Buffer::Format::kRGB:
    case Buffer::Format::kGRAY:
      return CropResize(buffer, x0, y0, x1, y1, output_buffer);
    case Buffer::Format::kNV12:
    case Buffer::Format::kNV21:
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21:
      return CropResizeYuv(buffer, x0, y0, x1, y1, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

absl::Status LibyuvBufferUtils::Resize(const Buffer& buffer,
                                       Buffer* output_buffer) {
  RETURN_IF_ERROR(ValidateResizeBufferInputs(buffer, *output_buffer));
  switch (buffer.format()) {
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21:
      return ResizeYv(buffer, output_buffer);
    case Buffer::Format::kNV12:
    case Buffer::Format::kNV21:
      return ResizeNv(buffer, output_buffer);
    case Buffer::Format::kRGB:
      return ResizeRgb(buffer, output_buffer);
    case Buffer::Format::kRGBA:
      return ResizeRgba(buffer, output_buffer);
    case Buffer::Format::kGRAY:
      return ResizeGray(buffer, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

absl::Status LibyuvBufferUtils::Rotate(const Buffer& buffer, int angle_deg,
                                       Buffer* output_buffer) {
  RETURN_IF_ERROR(
      ValidateRotateBufferInputs(buffer, *output_buffer, angle_deg));
  RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(buffer));
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(*output_buffer));

  switch (buffer.format()) {
    case Buffer::Format::kGRAY:
      return RotateGray(buffer, angle_deg, output_buffer);
    case Buffer::Format::kRGBA:
      return RotateRgba(buffer, angle_deg, output_buffer);
    case Buffer::Format::kNV12:
    case Buffer::Format::kNV21:
      return RotateNv(buffer, angle_deg, output_buffer);
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21:
      return RotateYv(buffer, angle_deg, output_buffer);
    case Buffer::Format::kRGB:
      return RotateRgb(buffer, angle_deg, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

absl::Status LibyuvBufferUtils::FlipHorizontally(const Buffer& buffer,
                                                 Buffer* output_buffer) {
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(buffer));
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(*output_buffer));
  RETURN_IF_ERROR(ValidateFlipBufferInputs(buffer, *output_buffer));
  RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case Buffer::Format::kRGBA:
      return FlipHorizontallyRgba(buffer, output_buffer);
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21:
      return FlipHorizontallyYv(buffer, output_buffer);
    case Buffer::Format::kNV12:
    case Buffer::Format::kNV21:
      return FlipHorizontallyNv(buffer, output_buffer);
    case Buffer::Format::kRGB:
      return FlipHorizontallyRgb(buffer, output_buffer);
    case Buffer::Format::kGRAY:
      return FlipHorizontallyPlane(buffer, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

absl::Status LibyuvBufferUtils::FlipVertically(const Buffer& buffer,
                                               Buffer* output_buffer) {
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(buffer));
  RETURN_IF_ERROR(ValidateBufferPlaneMetadata(*output_buffer));
  RETURN_IF_ERROR(ValidateFlipBufferInputs(buffer, *output_buffer));
  RETURN_IF_ERROR(ValidateBufferFormats(buffer, *output_buffer));

  switch (buffer.format()) {
    case Buffer::Format::kRGBA:
    case Buffer::Format::kRGB:
    case Buffer::Format::kGRAY:
      return FlipPlaneVertically(buffer, output_buffer);
    case Buffer::Format::kNV12:
    case Buffer::Format::kNV21:
      return FlipVerticallyNv(buffer, output_buffer);
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21:
      return FlipVerticallyYv(buffer, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

absl::Status LibyuvBufferUtils::Convert(const Buffer& buffer,
                                        Buffer* output_buffer) {
  RETURN_IF_ERROR(
      ValidateConvertFormats(buffer.format(), output_buffer->format()));
  switch (buffer.format()) {
    case Buffer::Format::kNV12:
      return ConvertFromNv12(buffer, output_buffer);
    case Buffer::Format::kNV21:
      return ConvertFromNv21(buffer, output_buffer);
    case Buffer::Format::kYV12:
    case Buffer::Format::kYV21:
      return ConvertFromYv(buffer, output_buffer);
    case Buffer::Format::kRGB:
      return ConvertFromRgb(buffer, output_buffer);
    case Buffer::Format::kRGBA:
      return ConvertFromRgba(buffer, output_buffer);
    default:
      return CreateStatusWithPayload(
          StatusCode::kInternal,
          absl::StrFormat("Format %i is not supported.", buffer.format()),
          TfLiteSupportStatus::kImageProcessingError);
  }
}

}  // namespace tensor
}  // namespace band
