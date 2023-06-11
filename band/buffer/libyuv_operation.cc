/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
Heavily modified from the original source code:
tflite-support/tensorflow_lite_support/cc/task/vision/utils
/libyuv_frame_buffer_utils.cc
by Jingyu Lee <dostos10@gmail.com>
*/

#include "band/buffer/libyuv_operation.h"

#include <stdint.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "band/buffer/buffer.h"
#include "band/common.h"
#include "libyuv.h"

namespace band {

using ::absl::StatusCode;

struct YuvData {
  size_t y_row_stride;
  size_t uv_row_stride;
  size_t uv_pixel_stride;
  const unsigned char* y_buffer;
  const unsigned char* u_buffer;
  const unsigned char* v_buffer;
};

absl::StatusOr<YuvData> GetYuvDataFromBuffer(const Buffer& buffer) {
  if (buffer.GetBufferFormat() != BufferFormat::kYV12 &&
      buffer.GetBufferFormat() != BufferFormat::kYV21 &&
      buffer.GetBufferFormat() != BufferFormat::kNV12 &&
      buffer.GetBufferFormat() != BufferFormat::kNV21) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Buffer format %i is not supported.", buffer.GetBufferFormat()));
  }

  if (buffer.GetNumPlanes() != 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Buffer with %i planes is not supported.", buffer.GetNumPlanes()));
  }

  YuvData yuv_data;
  if (buffer.GetBufferFormat() == BufferFormat::kNV21 ||
      buffer.GetBufferFormat() == BufferFormat::kYV12) {
    // Y follow by VU order. The VU chroma planes can be interleaved or
    // planar.
    yuv_data.y_buffer = buffer[0].data;
    yuv_data.v_buffer = buffer[1].data;
    yuv_data.u_buffer = buffer[2].data;
    yuv_data.y_row_stride = buffer[0].row_stride_bytes;
    yuv_data.uv_row_stride = buffer[1].row_stride_bytes;
    yuv_data.uv_pixel_stride = buffer[1].pixel_stride_bytes;
  } else {
    // Y follow by UV order. The UV chroma planes can be interleaved or
    // planar.
    yuv_data.y_buffer = buffer[0].data;
    yuv_data.u_buffer = buffer[1].data;
    yuv_data.v_buffer = buffer[2].data;
    yuv_data.y_row_stride = buffer[0].row_stride_bytes;
    yuv_data.uv_row_stride = buffer[1].row_stride_bytes;
    yuv_data.uv_pixel_stride = buffer[1].pixel_stride_bytes;
  }

  return yuv_data;
}

// Converts NV12 `buffer` to the `output_buffer` of the target color space.
// Supported output format includes RGB24 and YV21.
absl::Status ConvertFromNv12(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> yuv_data = GetYuvDataFromBuffer(buffer);
  if (!yuv_data.ok()) {
    return yuv_data.status();
  }
  switch (output_buffer.GetBufferFormat()) {
    case BufferFormat::kRGB: {
      // The RAW format of Libyuv represents the 8-bit interleaved RGB format in
      // the big endian style with R being the first byte in memory.
      int ret =
          libyuv::NV12ToRAW(yuv_data->y_buffer, yuv_data->y_row_stride,
                            yuv_data->u_buffer, yuv_data->uv_row_stride,
                            const_cast<unsigned char*>(output_buffer[0].data),
                            output_buffer[0].row_stride_bytes,
                            buffer.GetDimension()[0], buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12ToRAW operation failed.");
      }
      break;
    }
    case BufferFormat::kRGBA: {
      // The libyuv ABGR format is interleaved RGBA format in memory.
      int ret = libyuv::NV12ToABGR(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->u_buffer,
          yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_buffer[0].data),
          output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12ToABGR operation failed.");
      }
      break;
    }
    case BufferFormat::kYV12:
    case BufferFormat::kYV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::NV12ToI420(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->u_buffer,
          yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, output_buffer.GetDimension()[0],
          output_buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12ToI420 operation failed.");
      }
      break;
    }
    case BufferFormat::kNV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      libyuv::CopyPlane(yuv_data->y_buffer, yuv_data->y_row_stride,
                        const_cast<unsigned char*>(output_data->y_buffer),
                        output_data->y_row_stride, buffer.GetDimension()[0],
                        buffer.GetDimension()[1]);
      const std::vector<size_t> uv_plane_dimension = Buffer::GetUvDims(
          output_buffer.GetDimension(), output_buffer.GetBufferFormat());
      libyuv::SwapUVPlane(yuv_data->u_buffer, yuv_data->uv_row_stride,
                          const_cast<unsigned char*>(output_data->v_buffer),
                          output_data->uv_row_stride, uv_plane_dimension[0],
                          uv_plane_dimension[1]);
      break;
    }
    case BufferFormat::kGrayScale: {
      libyuv::CopyPlane(yuv_data->y_buffer, yuv_data->y_row_stride,
                        const_cast<unsigned char*>(output_buffer[0].data),
                        output_buffer[0].row_stride_bytes,
                        output_buffer.GetDimension()[0],
                        output_buffer.GetDimension()[1]);
      break;
    }
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %i is not supported.", output_buffer.GetBufferFormat()));
  }
  return absl::OkStatus();
}

// Converts NV21 `buffer` into the `output_buffer` of the target color space.
// Supported output format includes RGB24 and YV21.
absl::Status ConvertFromNv21(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> yuv_data = GetYuvDataFromBuffer(buffer);
  if (!yuv_data.ok()) {
    return yuv_data.status();
  }
  switch (output_buffer.GetBufferFormat()) {
    case BufferFormat::kRGB: {
      // The RAW format of Libyuv represents the 8-bit interleaved RGB format in
      // the big endian style with R being the first byte in memory.
      int ret =
          libyuv::NV21ToRAW(yuv_data->y_buffer, yuv_data->y_row_stride,
                            yuv_data->v_buffer, yuv_data->uv_row_stride,
                            const_cast<unsigned char*>(output_buffer[0].data),
                            output_buffer[0].row_stride_bytes,
                            buffer.GetDimension()[0], buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV21ToRAW operation failed.");
      }
      break;
    }
    case BufferFormat::kRGBA: {
      // The libyuv ABGR format is interleaved RGBA format in memory.
      int ret = libyuv::NV21ToABGR(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->v_buffer,
          yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_buffer[0].data),
          output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV21ToABGR operation failed.");
      }
      break;
    }
    case BufferFormat::kYV12:
    case BufferFormat::kYV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::NV21ToI420(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->v_buffer,
          yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, output_buffer.GetDimension()[0],
          output_buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV21ToI420 operation failed.");
      }
      break;
    }
    case BufferFormat::kNV12: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      libyuv::CopyPlane(yuv_data->y_buffer, yuv_data->y_row_stride,
                        const_cast<unsigned char*>(output_data->y_buffer),
                        output_data->y_row_stride, buffer.GetDimension()[0],
                        buffer.GetDimension()[1]);
      const std::vector<size_t> uv_plane_dimension = Buffer::GetUvDims(
          output_buffer.GetDimension(), output_buffer.GetBufferFormat());
      libyuv::SwapUVPlane(yuv_data->v_buffer, yuv_data->uv_row_stride,
                          const_cast<unsigned char*>(output_data->u_buffer),
                          output_data->uv_row_stride, uv_plane_dimension[0],
                          uv_plane_dimension[1]);
      break;
    }
    case BufferFormat::kGrayScale: {
      libyuv::CopyPlane(yuv_data->y_buffer, yuv_data->y_row_stride,
                        const_cast<unsigned char*>(output_buffer[0].data),
                        output_buffer[0].row_stride_bytes,
                        output_buffer.GetDimension()[0],
                        output_buffer.GetDimension()[1]);
      break;
    }
    default:
      return absl::InternalError(
          absl::StrFormat("Format %s is not supported.",
                          ToString(output_buffer.GetBufferFormat())));
  }
  return absl::OkStatus();
}

// Converts YV12/YV21 `buffer` to the `output_buffer` of the target color space.
// Supported output format includes RGB24, NV12, and NV21.
absl::Status ConvertFromYv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> yuv_data = GetYuvDataFromBuffer(buffer);
  if (!yuv_data.ok()) {
    return yuv_data.status();
  }
  switch (output_buffer.GetBufferFormat()) {
    case BufferFormat::kRGB: {
      // The RAW format of Libyuv represents the 8-bit interleaved RGB format in
      // the big endian style with R being the first byte in memory.
      int ret = libyuv::I420ToRAW(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->u_buffer,
          yuv_data->uv_row_stride, yuv_data->v_buffer, yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_buffer[0].data),
          output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToRAW operation failed.");
      }
      break;
    }
    case BufferFormat::kRGBA: {
      // The libyuv ABGR format is interleaved RGBA format in memory.
      int ret = libyuv::I420ToABGR(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->u_buffer,
          yuv_data->uv_row_stride, yuv_data->v_buffer, yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_buffer[0].data),
          output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToABGR operation failed.");
      }
      break;
    }
    case BufferFormat::kNV12: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::I420ToNV12(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->u_buffer,
          yuv_data->uv_row_stride, yuv_data->v_buffer, yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride, output_buffer.GetDimension()[0],
          output_buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToNV12 operation failed.");
      }
      break;
    }
    case BufferFormat::kNV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::I420ToNV21(
          yuv_data->y_buffer, yuv_data->y_row_stride, yuv_data->u_buffer,
          yuv_data->uv_row_stride, yuv_data->v_buffer, yuv_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, output_buffer.GetDimension()[0],
          output_buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420ToNV21 operation failed.");
      }
      break;
    }
    case BufferFormat::kGrayScale: {
      libyuv::CopyPlane(yuv_data->y_buffer, yuv_data->y_row_stride,
                        const_cast<unsigned char*>(output_buffer[0].data),
                        output_buffer[0].row_stride_bytes,
                        output_buffer.GetDimension()[0],
                        output_buffer.GetDimension()[1]);
      break;
    }
    case BufferFormat::kYV12:
    case BufferFormat::kYV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      std::vector<size_t> uv_plane_dimension = Buffer::GetUvDims(
          output_buffer.GetDimension(), output_buffer.GetBufferFormat());
      libyuv::CopyPlane(yuv_data->y_buffer, yuv_data->y_row_stride,
                        const_cast<unsigned char*>(output_data->y_buffer),
                        output_data->y_row_stride, buffer.GetDimension()[0],
                        buffer.GetDimension()[1]);
      libyuv::CopyPlane(yuv_data->u_buffer, yuv_data->uv_row_stride,
                        const_cast<unsigned char*>(output_data->u_buffer),
                        output_data->uv_row_stride, uv_plane_dimension[0],
                        uv_plane_dimension[1]);
      libyuv::CopyPlane(yuv_data->v_buffer, yuv_data->uv_row_stride,
                        const_cast<unsigned char*>(output_data->v_buffer),
                        output_data->uv_row_stride, uv_plane_dimension[0],
                        uv_plane_dimension[1]);
      break;
    }
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %i is not supported.", output_buffer.GetBufferFormat()));
  }
  return absl::OkStatus();
}

// Resizes YV12/YV21 `buffer` to the target `output_buffer`.
absl::Status ResizeYv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  // TODO(b/151217096): Choose the optimal image resizing filter to optimize
  // the model inference performance.
  int ret = libyuv::I420Scale(
      input_data->y_buffer, input_data->y_row_stride, input_data->u_buffer,
      input_data->uv_row_stride, input_data->v_buffer,
      input_data->uv_row_stride, buffer.GetDimension()[0],
      buffer.GetDimension()[1],
      const_cast<unsigned char*>(output_data->y_buffer),
      output_data->y_row_stride,
      const_cast<unsigned char*>(output_data->u_buffer),
      output_data->uv_row_stride,
      const_cast<unsigned char*>(output_data->v_buffer),
      output_data->uv_row_stride, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1], libyuv::FilterMode::kFilterBilinear);
  if (ret != 0) {
    return absl::UnknownError("Libyuv I420Scale operation failed.");
  }
  return absl::OkStatus();
}

// Resizes NV12/NV21 `buffer` to the target `output_buffer`.
absl::Status ResizeNv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  const unsigned char* src_uv = input_data->u_buffer;
  const unsigned char* dst_uv = output_data->u_buffer;
  if (buffer.GetBufferFormat() == BufferFormat::kNV21) {
    src_uv = input_data->v_buffer;
    dst_uv = output_data->v_buffer;
  }

  int ret = libyuv::NV12Scale(
      input_data->y_buffer, input_data->y_row_stride, src_uv,
      input_data->uv_row_stride, buffer.GetDimension()[0],
      buffer.GetDimension()[1],
      const_cast<unsigned char*>(output_data->y_buffer),
      output_data->y_row_stride, const_cast<unsigned char*>(dst_uv),
      output_data->uv_row_stride, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1], libyuv::FilterMode::kFilterBilinear);

  if (ret != 0) {
    return absl::UnknownError("Libyuv NV12Scale operation failed.");
  }
  return absl::OkStatus();
}

// Converts `buffer` to libyuv ARGB format and stores the conversion result
// in `dest_argb`.
absl::Status ConvertRgbToArgb(const Buffer& buffer, unsigned char* dest_argb,
                              int dest_stride_argb) {
  if (buffer.GetBufferFormat() != BufferFormat::kRGB) {
    return absl::InternalError("RGB input format is expected.");
  }

  if (dest_argb == nullptr || dest_stride_argb <= 0) {
    return absl::InternalError(
        "Invalid destination arguments for ConvertRgbToArgb.");
  }

  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }
  int ret = libyuv::RGB24ToARGB(
      buffer[0].data, buffer[0].row_stride_bytes, dest_argb, dest_stride_argb,
      buffer.GetDimension()[0], buffer.GetDimension()[1]);
  if (ret != 0) {
    return absl::UnknownError("Libyuv RGB24ToARGB operation failed.");
  }
  return absl::OkStatus();
}

// Converts `src_argb` in libyuv ARGB format to Buffer::kRGB format and
// stores the conversion result in `output_buffer`.
absl::Status ConvertArgbToRgb(unsigned char* src_argb, int src_stride_argb,
                              Buffer& output_buffer) {
  if (output_buffer.GetBufferFormat() != BufferFormat::kRGB) {
    return absl::InternalError("RGB input format is expected.");
  }

  if (src_argb == nullptr || src_stride_argb <= 0) {
    return absl::InternalError(
        "Invalid source arguments for ConvertArgbToRgb.");
  }

  if (output_buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        output_buffer.GetBufferFormat()));
  }
  int ret = libyuv::ARGBToRGB24(
      src_argb, src_stride_argb,
      const_cast<unsigned char*>(output_buffer[0].data),
      output_buffer[0].row_stride_bytes, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1]);

  if (ret != 0) {
    return absl::UnknownError("Libyuv ARGBToRGB24 operation failed.");
  }
  return absl::OkStatus();
}

// Converts `buffer` in Buffer::kRGBA format to libyuv ARGB (BGRA in
// memory) format and stores the conversion result in `dest_argb`.
absl::Status ConvertRgbaToArgb(const Buffer& buffer, unsigned char* dest_argb,
                               int dest_stride_argb) {
  if (buffer.GetBufferFormat() != BufferFormat::kRGBA) {
    return absl::InternalError("RGBA input format is expected.");
  }

  if (dest_argb == nullptr || dest_stride_argb <= 0) {
    return absl::InternalError(
        "Invalid source arguments for ConvertRgbaToArgb.");
  }

  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

  int ret = libyuv::ABGRToARGB(
      buffer[0].data, buffer[0].row_stride_bytes, dest_argb, dest_stride_argb,
      buffer.GetDimension()[0], buffer.GetDimension()[1]);
  if (ret != 0) {
    return absl::InternalError("Libyuv ABGRToARGB operation failed.");
  }
  return absl::OkStatus();
}

// Converts kRGB `buffer` to the `output_buffer` of the target color space.
absl::Status ConvertFromRgb(const Buffer& buffer, Buffer& output_buffer) {
  if (output_buffer.GetBufferFormat() == BufferFormat::kGrayScale) {
    int ret =
        libyuv::RAWToJ400(buffer[0].data, buffer[0].row_stride_bytes,
                          const_cast<unsigned char*>(output_buffer[0].data),
                          output_buffer[0].row_stride_bytes,
                          buffer.GetDimension()[0], buffer.GetDimension()[1]);
    if (ret != 0) {
      return absl::InternalError("Libyuv RAWToJ400 operation failed.");
    }
    return absl::OkStatus();
  } else if (output_buffer.GetBufferFormat() == BufferFormat::kYV12 ||
             output_buffer.GetBufferFormat() == BufferFormat::kYV21 ||
             output_buffer.GetBufferFormat() == BufferFormat::kNV12 ||
             output_buffer.GetBufferFormat() == BufferFormat::kNV21) {
    // libyuv does not support conversion directly from kRGB to kNV12 / kNV21.
    // For kNV12 / kNV21, the implementation converts the kRGB to I420,
    // then converts I420 to kNV12 / kNV21.
    // TODO(b/153000936): use libyuv::RawToNV12 / libyuv::RawToNV21 when they
    // are ready.
    absl::StatusOr<YuvData> yuv_data;
    std::unique_ptr<unsigned char[]> tmp_yuv_buffer;
    std::shared_ptr<Buffer> yuv_frame_buffer;
    if (output_buffer.GetBufferFormat() == BufferFormat::kNV12 ||
        output_buffer.GetBufferFormat() == BufferFormat::kNV21) {
      tmp_yuv_buffer =
          absl::make_unique<unsigned char[]>(Buffer::GetBufferByteSize(
              buffer.GetDimension(), output_buffer.GetBufferFormat()));

      yuv_frame_buffer = std::shared_ptr<Buffer>(
          Buffer::CreateFromRaw(tmp_yuv_buffer.get(), buffer.GetDimension()[0],
                                buffer.GetDimension()[1], BufferFormat::kYV21,
                                buffer.GetOrientation()));
      if (!yuv_frame_buffer) {
        return absl::InternalError("Failed to create YV21 buffer.");
      }

      yuv_data = GetYuvDataFromBuffer(*yuv_frame_buffer);
      if (!yuv_data.ok()) {
        return yuv_data.status();
      }
    } else {
      yuv_data = GetYuvDataFromBuffer(output_buffer);
      if (!yuv_data.ok()) {
        return yuv_data.status();
      }
    }
    int ret = libyuv::RAWToI420(
        buffer[0].data, buffer[0].row_stride_bytes,
        const_cast<unsigned char*>(yuv_data->y_buffer), yuv_data->y_row_stride,
        const_cast<unsigned char*>(yuv_data->u_buffer), yuv_data->uv_row_stride,
        const_cast<unsigned char*>(yuv_data->v_buffer), yuv_data->uv_row_stride,
        buffer.GetDimension()[0], buffer.GetDimension()[1]);
    if (ret != 0) {
      return absl::InternalError("Libyuv RAWToI420 operation failed.");
    }
    if (output_buffer.GetBufferFormat() == BufferFormat::kNV12 ||
        output_buffer.GetBufferFormat() == BufferFormat::kNV21) {
      return ConvertFromYv(*yuv_frame_buffer, output_buffer);
    }
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat("Format %i is not supported.",
                                             output_buffer.GetBufferFormat()));
}

// Converts kRGBA `buffer` to the `output_buffer` of the target color space.
absl::Status ConvertFromRgba(const Buffer& buffer, Buffer& output_buffer) {
  switch (output_buffer.GetBufferFormat()) {
    case BufferFormat::kGrayScale: {
      // libyuv does not support convert kRGBA (ABGR) foramat. In this method,
      // the implementation converts kRGBA format to ARGB and use ARGB buffer
      // for conversion.
      // TODO(b/141181395): Use libyuv::ABGRToJ400 when it is ready.

      // Convert kRGBA to ARGB
      int argb_buffer_size =
          Buffer::GetBufferByteSize(buffer.GetDimension(), BufferFormat::kRGBA);
      auto argb_buffer = absl::make_unique<unsigned char[]>(argb_buffer_size);
      const int argb_row_bytes = buffer.GetDimension()[0] * 4;

      absl::Status status =
          ConvertRgbaToArgb(buffer, argb_buffer.get(), argb_row_bytes);
      if (!status.ok()) {
        return status;
      }

      // Convert ARGB to kGRAY
      int ret = libyuv::ARGBToJ400(
          argb_buffer.get(), argb_row_bytes,
          const_cast<unsigned char*>(output_buffer[0].data),
          output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ARGBToJ400 operation failed.");
      }
      break;
    }
    case BufferFormat::kNV12: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::ABGRToNV12(
          buffer[0].data, buffer[0].row_stride_bytes,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToNV12 operation failed.");
      }
      break;
    }
    case BufferFormat::kNV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::ABGRToNV21(
          buffer[0].data, buffer[0].row_stride_bytes,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToNV21 operation failed.");
      }
      break;
    }
    case BufferFormat::kYV12:
    case BufferFormat::kYV21: {
      absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
      if (!output_data.ok()) {
        return output_data.status();
      }
      int ret = libyuv::ABGRToI420(
          buffer[0].data, buffer[0].row_stride_bytes,
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToI420 operation failed.");
      }
      break;
    }
    case BufferFormat::kRGB: {
      // ARGB is BGRA in memory and RGB24 is BGR in memory. The removal of the
      // alpha channel will not impact the RGB ordering.
      int ret = libyuv::ARGBToRGB24(
          buffer[0].data, buffer[0].row_stride_bytes,
          const_cast<unsigned char*>(output_buffer[0].data),
          output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
          buffer.GetDimension()[1]);
      if (ret != 0) {
        return absl::UnknownError("Libyuv ABGRToRGB24 operation failed.");
      }
      break;
    }
    default:
      return absl::InternalError(
          absl::StrFormat("Convert Rgba to format  %s is not supported.",
                          ToString(output_buffer.GetBufferFormat())));
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
                        Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

  // libyuv::ARGBRotate assumes RGBA buffer is in the interleaved format.
  int ret = libyuv::ARGBRotate(
      buffer[0].data, buffer[0].row_stride_bytes,
      const_cast<unsigned char*>(output_buffer[0].data),
      output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
      buffer.GetDimension()[1], GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return absl::UnknownError("Libyuv ARGBRotate operation failed.");
  }
  return absl::OkStatus();
}

absl::Status RotateRgb(const Buffer& buffer, int angle_deg,
                       Buffer& output_buffer) {
  // libyuv does not support rotate kRGB (RGB24) foramat. In this method, the
  // implementation converts kRGB format to ARGB and use ARGB buffer for
  // rotation. The result is then convert back to RGB.

  // Convert RGB to ARGB
  int argb_buffer_size =
      Buffer::GetBufferByteSize(buffer.GetDimension(), BufferFormat::kRGBA);
  auto argb_buffer = absl::make_unique<unsigned char[]>(argb_buffer_size);
  const int argb_row_bytes = buffer.GetDimension()[0] * 4;

  absl::Status status =
      ConvertRgbToArgb(buffer, argb_buffer.get(), argb_row_bytes);
  if (!status.ok()) {
    return status;
  }

  // Rotate ARGB
  auto argb_rotated_buffer =
      absl::make_unique<unsigned char[]>(argb_buffer_size);
  int rotated_row_bytes = output_buffer.GetDimension()[0] * 4;
  // TODO(b/151954340): Optimize the current implementation by utilizing
  // ARGBMirror for 180 degree rotation.
  int ret = libyuv::ARGBRotate(
      argb_buffer.get(), argb_row_bytes, argb_rotated_buffer.get(),
      rotated_row_bytes, buffer.GetDimension()[0], buffer.GetDimension()[1],
      GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return absl::UnknownError("Libyuv ARGBRotate operation failed.");
  }

  // Convert ARGB to RGB
  return ConvertArgbToRgb(argb_rotated_buffer.get(), rotated_row_bytes,
                          output_buffer);
}

absl::Status RotateGray(const Buffer& buffer, int angle_deg,
                        Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }
  int ret = libyuv::RotatePlane(
      buffer[0].data, buffer[0].row_stride_bytes,
      const_cast<unsigned char*>(output_buffer[0].data),
      output_buffer[0].row_stride_bytes, buffer.GetDimension()[0],
      buffer.GetDimension()[1], GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return absl::UnknownError("Libyuv RotatePlane operation failed.");
  }
  return absl::OkStatus();
}

// Rotates YV12/YV21 frame buffer.
absl::Status RotateYv(const Buffer& buffer, int angle_deg,
                      Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  int ret = libyuv::I420Rotate(
      input_data->y_buffer, input_data->y_row_stride, input_data->u_buffer,
      input_data->uv_row_stride, input_data->v_buffer,
      input_data->uv_row_stride,
      const_cast<unsigned char*>(output_data->y_buffer),
      output_data->y_row_stride,
      const_cast<unsigned char*>(output_data->u_buffer),
      output_data->uv_row_stride,
      const_cast<unsigned char*>(output_data->v_buffer),
      output_data->uv_row_stride, buffer.GetDimension()[0],
      buffer.GetDimension()[1], GetLibyuvRotationMode(angle_deg));
  if (ret != 0) {
    return absl::UnknownError("Libyuv I420Rotate operation failed.");
  }
  return absl::OkStatus();
}

// Rotates NV12/NV21 frame buffer.
// TODO(b/152097364): Refactor NV12/NV21 rotation after libyuv explicitly
// support that.
absl::Status RotateNv(const Buffer& buffer, int angle_deg,
                      Buffer& output_buffer) {
  if (buffer.GetBufferFormat() != BufferFormat::kNV12 &&
      buffer.GetBufferFormat() != BufferFormat::kNV21) {
    return absl::InternalError("kNV12 or kNV21 input formats are expected.");
  }
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  const int rotated_buffer_size = Buffer::GetBufferByteSize(
      output_buffer.GetDimension(), BufferFormat::kYV21);
  auto rotated_yuv_raw_buffer =
      absl::make_unique<unsigned char[]>(rotated_buffer_size);
  std::shared_ptr<Buffer> rotated_yuv_buffer(Buffer::CreateFromRaw(
      rotated_yuv_raw_buffer.get(), output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1], BufferFormat::kYV21,
      output_buffer.GetOrientation()));
  if (!rotated_yuv_buffer) {
    return absl::InternalError("Failed to create YV21 buffer.");
  }
  absl::StatusOr<YuvData> rotated_yuv_data =
      GetYuvDataFromBuffer(*rotated_yuv_buffer);
  if (!rotated_yuv_data.ok()) {
    return rotated_yuv_data.status();
  }
  // Get the first chroma plane and use it as the u plane. This is a workaround
  // for optimizing NV21 rotation. For NV12, the implementation is logical
  // correct. For NV21, use v plane as u plane will make the UV planes swapped
  // in the intermediate rotated I420 frame. The output buffer is finally built
  // by merging the swapped UV planes which produces V first interleaved UV
  // buffer.
  const unsigned char* chroma_buffer =
      buffer.GetBufferFormat() == BufferFormat::kNV12 ? input_data->u_buffer
                                                      : input_data->v_buffer;
  // Rotate the Y plane and store into the Y plane in `output_buffer`. Rotate
  // the interleaved UV plane and store into the interleaved UV plane in
  // `rotated_yuv_buffer`.
  int ret = libyuv::NV12ToI420Rotate(
      input_data->y_buffer, input_data->y_row_stride, chroma_buffer,
      input_data->uv_row_stride,
      const_cast<unsigned char*>(output_data->y_buffer),
      output_data->y_row_stride,
      const_cast<unsigned char*>(rotated_yuv_data->u_buffer),
      rotated_yuv_data->uv_row_stride,
      const_cast<unsigned char*>(rotated_yuv_data->v_buffer),
      rotated_yuv_data->uv_row_stride, buffer.GetDimension()[0],
      buffer.GetDimension()[1], GetLibyuvRotationMode(angle_deg % 360));
  if (ret != 0) {
    return absl::UnknownError("Libyuv Nv12ToI420Rotate operation failed.");
  }
  // Merge rotated UV planes into the output buffer. For NV21, the UV buffer of
  // the intermediate I420 frame is swapped. MergeUVPlane builds the interleaved
  // VU buffer for NV21 by putting the U plane in the I420 frame which is
  // actually the V plane from the input buffer first.
  const unsigned char* output_chroma_buffer =
      buffer.GetBufferFormat() == BufferFormat::kNV12 ? output_data->u_buffer
                                                      : output_data->v_buffer;
  // The width and height arguments of `libyuv::MergeUVPlane()` represent the
  // width and height of the UV planes.
  libyuv::MergeUVPlane(
      rotated_yuv_data->u_buffer, rotated_yuv_data->uv_row_stride,
      rotated_yuv_data->v_buffer, rotated_yuv_data->uv_row_stride,
      const_cast<unsigned char*>(output_chroma_buffer),
      output_data->uv_row_stride, (output_buffer.GetDimension()[0] + 1) / 2,
      (output_buffer.GetDimension()[1] + 1) / 2);
  return absl::OkStatus();
}

// This method only supports kGRAY, kRGB, and kRGBA format.
absl::Status FlipPlaneVertically(const Buffer& buffer, Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

  size_t pixel_stride = buffer.GetPixelBytes();

  // Flip vertically is achieved by passing in negative height.
  libyuv::CopyPlane(buffer[0].data, buffer[0].row_stride_bytes,
                    const_cast<unsigned char*>(output_buffer[0].data),
                    output_buffer[0].row_stride_bytes,
                    output_buffer.GetDimension()[0] * pixel_stride,
                    -output_buffer.GetDimension()[1]);

  return absl::OkStatus();
}

// This method only supports kGRAY, kRGBA, and kRGB formats.
absl::Status CropPlane(const Buffer& buffer, int x0, int y0, int x1, int y1,
                       Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

  size_t pixel_stride = buffer.GetPixelBytes();
  std::vector<size_t> crop_dimension = Buffer::GetCropDimension(x0, x1, y0, y1);

  // Cropping is achieved by adjusting origin to (x0, y0).
  int adjusted_offset = buffer[0].row_stride_bytes * y0 + x0 * pixel_stride;

  libyuv::CopyPlane(buffer[0].data + adjusted_offset,
                    buffer[0].row_stride_bytes,
                    const_cast<unsigned char*>(output_buffer[0].data),
                    output_buffer[0].row_stride_bytes,
                    crop_dimension[0] * pixel_stride, crop_dimension[1]);

  return absl::OkStatus();
}

const unsigned char* GetUvRawBuffer(const Buffer& buffer) {
  if (buffer.GetBufferFormat() != BufferFormat::kNV12 &&
      buffer.GetBufferFormat() != BufferFormat::kNV21) {
    return nullptr;
  }

  absl::StatusOr<YuvData> yuv_data = GetYuvDataFromBuffer(buffer);
  if (!yuv_data.ok()) {
    return nullptr;
  }

  return buffer.GetBufferFormat() == BufferFormat::kNV12 ? yuv_data->u_buffer
                                                         : yuv_data->v_buffer;
}

// Crops NV12/NV21 Buffer to the subregion defined by the top left pixel
// position (x0, y0) and the bottom right pixel position (x1, y1).
absl::Status CropNv(const Buffer& buffer, int x0, int y0, int x1, int y1,
                    Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  // Crop Y plane by copying the buffer with the origin offset to (x0, y0).
  int crop_offset_y = input_data->y_row_stride * y0 + x0;
  int crop_width = x1 - x0 + 1;
  int crop_height = y1 - y0 + 1;
  libyuv::CopyPlane(input_data->y_buffer + crop_offset_y,
                    input_data->y_row_stride,
                    const_cast<unsigned char*>(output_data->y_buffer),
                    output_data->y_row_stride, crop_width, crop_height);
  // Crop chroma plane by copying the buffer with the origin offset to
  // (x0 / 2, y0 / 2);
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  int crop_offset_chroma = input_data->uv_row_stride * (y0 / 2) +
                           input_data->uv_pixel_stride * (x0 / 2);

  const unsigned char* input_chroma_buffer = GetUvRawBuffer(buffer);
  if (input_chroma_buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get chroma buffer for format %i.",
                        buffer.GetBufferFormat()));
  };

  const unsigned char* output_chroma_buffer = GetUvRawBuffer(output_buffer);
  if (output_chroma_buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get chroma buffer for format %i.",
                        output_buffer.GetBufferFormat()));
  };

  libyuv::CopyPlane(
      input_chroma_buffer + crop_offset_chroma, input_data->uv_row_stride,
      const_cast<unsigned char*>(output_chroma_buffer),
      output_data->uv_row_stride,
      /*width=*/(crop_width + 1) / 2 * 2, /*height=*/(crop_height + 1) / 2);
  return absl::OkStatus();
}

// Crops YV12/YV21 Buffer to the subregion defined by the top left pixel
// position (x0, y0) and the bottom right pixel position (x1, y1).
absl::Status CropYv(const Buffer& buffer, int x0, int y0, int x1, int y1,
                    Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  // Crop Y plane by copying the buffer with the origin offset to (x0, y0).
  int crop_offset_y = input_data->y_row_stride * y0 + x0;
  std::vector<size_t> crop_dimension = Buffer::GetCropDimension(x0, x1, y0, y1);
  libyuv::CopyPlane(
      input_data->y_buffer + crop_offset_y, input_data->y_row_stride,
      const_cast<unsigned char*>(output_data->y_buffer),
      output_data->y_row_stride, crop_dimension[0], crop_dimension[1]);
  // Crop U plane by copying the buffer with the origin offset to
  // (x0 / 2, y0 / 2).
  const std::vector<size_t> crop_uv_dimension =
      Buffer::GetUvDims(crop_dimension, buffer.GetBufferFormat());
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  int crop_offset_chroma = input_data->uv_row_stride * (y0 / 2) +
                           input_data->uv_pixel_stride * (x0 / 2);
  libyuv::CopyPlane(
      input_data->u_buffer + crop_offset_chroma, input_data->uv_row_stride,
      const_cast<unsigned char*>(output_data->u_buffer),
      output_data->uv_row_stride, crop_uv_dimension[0], crop_uv_dimension[1]);
  // Crop V plane by copying the buffer with the origin offset to
  // (x0 / 2, y0 / 2);
  libyuv::CopyPlane(input_data->v_buffer + crop_offset_chroma,
                    input_data->uv_row_stride,
                    const_cast<unsigned char*>(output_data->v_buffer),
                    output_data->uv_row_stride,
                    /*width=*/(crop_dimension[0] + 1) / 2,
                    /*height=*/(crop_dimension[1] + 1) / 2);
  return absl::OkStatus();
}

absl::Status CropResizeYuv(const Buffer& buffer, int x0, int y0, int x1, int y1,
                           Buffer& output_buffer) {
  std::vector<size_t> crop_dimension = Buffer::GetCropDimension(x0, x1, y0, y1);
  if (crop_dimension == output_buffer.GetDimension()) {
    switch (buffer.GetBufferFormat()) {
      case BufferFormat::kNV12:
      case BufferFormat::kNV21:
        return CropNv(buffer, x0, y0, x1, y1, output_buffer);
      case BufferFormat::kYV12:
      case BufferFormat::kYV21:
        return CropYv(buffer, x0, y0, x1, y1, output_buffer);
      default:
        return absl::InternalError(absl::StrFormat(
            "Format %i is not supported.", buffer.GetBufferFormat()));
    }
  }
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }

  // Cropping YUV planes by offsetting the origins of each plane.
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  const int plane_y_offset = input_data->y_row_stride * y0 + x0;
  const int plane_uv_offset = input_data->uv_row_stride * (y0 / 2) +
                              input_data->uv_pixel_stride * (x0 / 2);
  Buffer::DataPlane cropped_plane_y = {input_data->y_buffer + plane_y_offset,
                                       input_data->y_row_stride,
                                       /*pixel_stride_bytes=*/1};
  Buffer::DataPlane cropped_plane_u = {input_data->u_buffer + plane_uv_offset,
                                       input_data->uv_row_stride,
                                       input_data->uv_pixel_stride};
  Buffer::DataPlane cropped_plane_v = {input_data->v_buffer + plane_uv_offset,
                                       input_data->uv_row_stride,
                                       input_data->uv_pixel_stride};

  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kNV12: {
      std::shared_ptr<Buffer> cropped_buffer(Buffer::CreateFromPlanes(
          {cropped_plane_y, cropped_plane_u, cropped_plane_v}, crop_dimension,
          buffer.GetBufferFormat(), buffer.GetOrientation()));
      return ResizeNv(*cropped_buffer, output_buffer);
    }
    case BufferFormat::kNV21: {
      std::shared_ptr<Buffer> cropped_buffer(Buffer::CreateFromPlanes(
          {cropped_plane_y, cropped_plane_v, cropped_plane_u}, crop_dimension,
          buffer.GetBufferFormat(), buffer.GetOrientation()));
      return ResizeNv(*cropped_buffer, output_buffer);
    }
    case BufferFormat::kYV12: {
      std::shared_ptr<Buffer> cropped_buffer(Buffer::CreateFromPlanes(
          {cropped_plane_y, cropped_plane_v, cropped_plane_u}, crop_dimension,
          buffer.GetBufferFormat(), buffer.GetOrientation()));
      return ResizeYv(*cropped_buffer, output_buffer);
    }
    case BufferFormat::kYV21: {
      std::shared_ptr<Buffer> cropped_buffer(Buffer::CreateFromPlanes(
          {cropped_plane_y, cropped_plane_u, cropped_plane_v}, crop_dimension,
          buffer.GetBufferFormat(), buffer.GetOrientation()));
      return ResizeYv(*cropped_buffer, output_buffer);
    }
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
  return absl::OkStatus();
}

absl::Status FlipHorizontallyRgba(const Buffer& buffer, Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

  int ret = libyuv::ARGBMirror(
      buffer[0].data, buffer[0].row_stride_bytes,
      const_cast<unsigned char*>(output_buffer[0].data),
      output_buffer[0].row_stride_bytes, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1]);

  if (ret != 0) {
    return absl::UnknownError("Libyuv ARGBMirror operation failed.");
  }

  return absl::OkStatus();
}

// Flips `buffer` horizontally and store the result in `output_buffer`. This
// method assumes all buffers have pixel stride equals to 1.
absl::Status FlipHorizontallyPlane(const Buffer& buffer,
                                   Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }
  libyuv::MirrorPlane(buffer[0].data, buffer[0].row_stride_bytes,
                      const_cast<unsigned char*>(output_buffer[0].data),
                      output_buffer[0].row_stride_bytes,
                      output_buffer.GetDimension()[0],
                      output_buffer.GetDimension()[1]);

  return absl::OkStatus();
}

absl::Status ResizeRgb(const Buffer& buffer, Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

  // libyuv doesn't support scale kRGB (RGB24) foramat. In this method,
  // the implementation converts kRGB format to ARGB and use ARGB buffer for
  // scaling. The result is then convert back to RGB.

  // Convert RGB to ARGB
  int argb_buffer_size =
      Buffer::GetBufferByteSize(buffer.GetDimension(), BufferFormat::kRGBA);
  auto argb_buffer = absl::make_unique<unsigned char[]>(argb_buffer_size);
  const int argb_row_bytes = buffer.GetDimension()[0] * 4;

  absl::Status status =
      ConvertRgbToArgb(buffer, argb_buffer.get(), argb_row_bytes);
  if (!status.ok()) {
    return status;
  }

  // Resize ARGB
  int resized_argb_buffer_size = Buffer::GetBufferByteSize(
      output_buffer.GetDimension(), BufferFormat::kRGBA);
  auto resized_argb_buffer =
      absl::make_unique<unsigned char[]>(resized_argb_buffer_size);
  int resized_argb_row_bytes = output_buffer.GetDimension()[0] * 4;
  int ret = libyuv::ARGBScale(
      argb_buffer.get(), argb_row_bytes, buffer.GetDimension()[0],
      buffer.GetDimension()[1], resized_argb_buffer.get(),
      resized_argb_row_bytes, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1], libyuv::FilterMode::kFilterBilinear);
  if (ret != 0) {
    return absl::UnknownError("Libyuv ARGBScale operation failed.");
  }

  // Convert ARGB to RGB
  return ConvertArgbToRgb(resized_argb_buffer.get(), resized_argb_row_bytes,
                          output_buffer);
}

// Horizontally flip `buffer` and store the result in `output_buffer`.
absl::Status FlipHorizontallyRgb(const Buffer& buffer, Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }

#if LIBYUV_VERSION >= 1747
  int ret =
      libyuv::RGB24Mirror(buffer[0].data, buffer[0].row_stride_bytes,
                          const_cast<unsigned char*>(output_buffer[0].data),
                          output_buffer[0].row_stride_bytes,
                          buffer.GetDimension()[0], buffer.GetDimension()[1]);
  if (ret != 0) {
    return absl::UnknownError("Libyuv RGB24Mirror operation failed.");
  }

  return absl::OkStatus();
#else
#error LibyuvBufferUtils requires LIBYUV_VERSION 1747 or above
#endif  // LIBYUV_VERSION >= 1747
}

absl::Status ResizeRgba(const Buffer& buffer, Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }
  int ret = libyuv::ARGBScale(
      buffer[0].data, buffer[0].row_stride_bytes, buffer.GetDimension()[0],
      buffer.GetDimension()[1],
      const_cast<unsigned char*>(output_buffer[0].data),
      output_buffer[0].row_stride_bytes, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1], libyuv::FilterMode::kFilterBilinear);
  if (ret != 0) {
    return absl::UnknownError("Libyuv ARGBScale operation failed.");
  }
  return absl::OkStatus();
}

// Flips NV12/NV21 Buffer horizontally.
absl::Status FlipHorizontallyNv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }

  const unsigned char* input_chroma_buffer = GetUvRawBuffer(buffer);
  if (input_chroma_buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get chroma buffer for format %i.",
                        buffer.GetBufferFormat()));
  };
  const unsigned char* output_chroma_buffer = GetUvRawBuffer(output_buffer);
  if (output_chroma_buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get chroma buffer for format %i.",
                        output_buffer.GetBufferFormat()));
  };

  int ret =
      libyuv::NV12Mirror(input_data->y_buffer, input_data->y_row_stride,
                         input_chroma_buffer, input_data->uv_row_stride,
                         const_cast<unsigned char*>(output_data->y_buffer),
                         output_data->y_row_stride,
                         const_cast<unsigned char*>(output_chroma_buffer),
                         output_data->uv_row_stride, buffer.GetDimension()[0],
                         buffer.GetDimension()[1]);

  if (ret != 0) {
    return absl::UnknownError("Libyuv NV12Mirror operation failed.");
  }

  return absl::OkStatus();
}

// Flips YV12/YV21 Buffer horizontally.
absl::Status FlipHorizontallyYv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  int ret =
      libyuv::I420Mirror(input_data->y_buffer, input_data->y_row_stride,
                         input_data->u_buffer, input_data->uv_row_stride,
                         input_data->v_buffer, input_data->uv_row_stride,
                         const_cast<unsigned char*>(output_data->y_buffer),
                         output_data->y_row_stride,
                         const_cast<unsigned char*>(output_data->u_buffer),
                         output_data->uv_row_stride,
                         const_cast<unsigned char*>(output_data->v_buffer),
                         output_data->uv_row_stride, buffer.GetDimension()[0],
                         buffer.GetDimension()[1]);
  if (ret != 0) {
    return absl::UnknownError("Libyuv I420Mirror operation failed.");
  }

  return absl::OkStatus();
}

// Flips NV12/NV21 Buffer vertically.
absl::Status FlipVerticallyNv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  // Flip Y plane vertically by passing a negative height.
  libyuv::CopyPlane(input_data->y_buffer, input_data->y_row_stride,
                    const_cast<unsigned char*>(output_data->y_buffer),
                    output_data->y_row_stride, buffer.GetDimension()[0],
                    -output_buffer.GetDimension()[1]);
  // Flip UV plane vertically by passing a negative height.
  const unsigned char* input_chroma_buffer = GetUvRawBuffer(buffer);
  if (input_chroma_buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get chroma buffer for format %i.",
                        buffer.GetBufferFormat()));
  };
  const unsigned char* output_chroma_buffer = GetUvRawBuffer(output_buffer);
  if (output_chroma_buffer == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get chroma buffer for format %i.",
                        output_buffer.GetBufferFormat()));
  };
  const std::vector<size_t> uv_plane_dimension =
      Buffer::GetUvDims(buffer.GetDimension(), buffer.GetBufferFormat());
  libyuv::CopyPlane(input_chroma_buffer, input_data->uv_row_stride,
                    const_cast<unsigned char*>(output_chroma_buffer),
                    output_data->uv_row_stride,
                    /*width=*/uv_plane_dimension[0] * 2,
                    -uv_plane_dimension[1]);
  return absl::OkStatus();
}

// Flips NV12/NV21 Buffer vertically.
absl::Status FlipVerticallyYv(const Buffer& buffer, Buffer& output_buffer) {
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);
  if (!output_data.ok()) {
    return output_data.status();
  }
  // Flip buffer vertically by passing a negative height.
  int ret =
      libyuv::I420Copy(input_data->y_buffer, input_data->y_row_stride,
                       input_data->u_buffer, input_data->uv_row_stride,
                       input_data->v_buffer, input_data->uv_row_stride,
                       const_cast<unsigned char*>(output_data->y_buffer),
                       output_data->y_row_stride,
                       const_cast<unsigned char*>(output_data->u_buffer),
                       output_data->uv_row_stride,
                       const_cast<unsigned char*>(output_data->v_buffer),
                       output_data->uv_row_stride, buffer.GetDimension()[0],
                       -buffer.GetDimension()[1]);
  if (ret != 0) {
    return absl::UnknownError("Libyuv I420Copy operation failed.");
  }
  return absl::OkStatus();
}

// Resize `buffer` to metadata defined in `output_buffer`. This
// method assumes buffer has pixel stride equals to 1 (grayscale equivalent).
absl::Status ResizeGray(const Buffer& buffer, Buffer& output_buffer) {
  if (buffer.GetNumPlanes() > 1) {
    return absl::InternalError(
        absl::StrFormat("Only single plane is supported for format %i.",
                        buffer.GetBufferFormat()));
  }
  libyuv::ScalePlane(
      buffer[0].data, buffer[0].row_stride_bytes, buffer.GetDimension()[0],
      buffer.GetDimension()[1],
      const_cast<unsigned char*>(output_buffer[0].data),
      output_buffer[0].row_stride_bytes, output_buffer.GetDimension()[0],
      output_buffer.GetDimension()[1], libyuv::FilterMode::kFilterBilinear);
  return absl::OkStatus();
}

// This method only supports kGRAY, kRGBA, and kRGB formats.
absl::Status CropResize(const Buffer& buffer, int x0, int y0, int x1, int y1,
                        Buffer& output_buffer) {
  std::vector<size_t> crop_dimension = Buffer::GetCropDimension(x0, x1, y0, y1);
  if (crop_dimension == output_buffer.GetDimension()) {
    return CropPlane(buffer, x0, y0, x1, y1, output_buffer);
  }

  size_t pixel_stride = buffer.GetPixelBytes();
  // Cropping is achieved by adjusting origin to (x0, y0).
  int adjusted_offset = buffer[0].row_stride_bytes * y0 + x0 * pixel_stride;
  Buffer::DataPlane plane = {buffer[0].data + adjusted_offset,
                             buffer[0].row_stride_bytes, pixel_stride};
  std::shared_ptr<Buffer> adjusted_buffer(Buffer::CreateFromPlanes(
      {plane}, crop_dimension, buffer.GetBufferFormat(),
      buffer.GetOrientation()));

  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kRGB:
      return ResizeRgb(*adjusted_buffer, output_buffer);
    case BufferFormat::kRGBA:
      return ResizeRgba(*adjusted_buffer, output_buffer);
    case BufferFormat::kGrayScale:
      return ResizeGray(*adjusted_buffer, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

// Returns the scaled dimension of the input_size best fit within the
// output_size bound while respecting the aspect ratio.
std::vector<size_t> GetScaledDimension(std::vector<size_t> input_size,
                                       std::vector<size_t> output_size) {
  size_t original_width = input_size[0];
  size_t original_height = input_size[1];
  size_t bound_width = output_size[0];
  size_t bound_height = output_size[1];
  size_t new_width = original_width;
  size_t new_height = original_height;

  // Try to fit the width first.
  new_width = bound_width;
  new_height = (new_width * original_height) / original_width;

  // Try to fit the height if needed.
  if (new_height > bound_height) {
    new_height = bound_height;
    new_width = (new_height * original_width) / original_height;
  }
  return std::vector<size_t>{new_width, new_height};
}

// This method only supports kGRAY, kRGBA, and kRGB formats.
absl::Status UniformCropResizePlane(const Buffer& buffer,
                                    std::vector<int> crop_coordinates,
                                    Buffer& output_buffer) {
  int x0 = 0, y0 = 0;
  std::vector<size_t> input_dimension = buffer.GetDimension();
  if (!crop_coordinates.empty()) {
    x0 = crop_coordinates[0];
    y0 = crop_coordinates[1];
    input_dimension = Buffer::GetCropDimension(x0, crop_coordinates[2], y0,
                                               crop_coordinates[3]);
  }
  if (input_dimension == output_buffer.GetDimension()) {
    // Cropping only case.
    return CropPlane(buffer, x0, y0, crop_coordinates[2], crop_coordinates[3],
                     output_buffer);
  }

  // Cropping is achieved by adjusting origin to (x0, y0).

  size_t pixel_stride = buffer.GetPixelBytes();
  int adjusted_offset = buffer[0].row_stride_bytes * y0 + x0 * pixel_stride;
  Buffer::DataPlane plane = {buffer[0].data + adjusted_offset,
                             buffer[0].row_stride_bytes, pixel_stride};
  std::shared_ptr<Buffer> adjusted_buffer(Buffer::CreateFromPlanes(
      {plane}, input_dimension, buffer.GetBufferFormat(),
      buffer.GetOrientation()));

  // Uniform resize is achieved by adjusting the resize dimension to fit the
  // output_buffer and respect the input aspect ratio at the same time. We
  // create an intermediate output buffer with adjusted dimension and point
  // its backing buffer to the output_buffer. Note the stride information on
  // the adjusted_output_buffer is not used in the Resize* methods.
  std::vector<size_t> adjusted_dimension =
      GetScaledDimension(input_dimension, output_buffer.GetDimension());
  Buffer::DataPlane output_plane = output_buffer[0];

  std::shared_ptr<Buffer> adjusted_output_buffer(Buffer::CreateFromPlanes(
      {output_plane}, adjusted_dimension, output_buffer.GetBufferFormat(),
      output_buffer.GetOrientation()));

  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kRGB:
      return ResizeRgb(*adjusted_buffer, *adjusted_output_buffer);
    case BufferFormat::kRGBA:
      return ResizeRgba(*adjusted_buffer, *adjusted_output_buffer);
    case BufferFormat::kGrayScale:
      return ResizeGray(*adjusted_buffer, *adjusted_output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

absl::Status UniformCropResizeYuv(const Buffer& buffer,
                                  std::vector<int> crop_coordinates,
                                  Buffer& output_buffer) {
  int x0 = 0, y0 = 0;
  std::vector<size_t> input_dimension = buffer.GetDimension();
  if (!crop_coordinates.empty()) {
    x0 = crop_coordinates[0];
    y0 = crop_coordinates[1];
    input_dimension = Buffer::GetCropDimension(x0, crop_coordinates[2], y0,
                                               crop_coordinates[3]);
  }
  if (input_dimension == output_buffer.GetDimension()) {
    // Cropping only case.
    int x1 = crop_coordinates[2];
    int y1 = crop_coordinates[3];
    switch (buffer.GetBufferFormat()) {
      case BufferFormat::kNV12:
      case BufferFormat::kNV21:
        return CropNv(buffer, x0, y0, x1, y1, output_buffer);
      case BufferFormat::kYV12:
      case BufferFormat::kYV21:
        return CropYv(buffer, x0, y0, x1, y1, output_buffer);
      default:
        return absl::InternalError(absl::StrFormat(
            "Format %i is not supported.", buffer.GetBufferFormat()));
    }
  }

  // Cropping is achieved by adjusting origin to (x0, y0).
  absl::StatusOr<YuvData> input_data = GetYuvDataFromBuffer(buffer);
  if (!input_data.ok()) {
    return input_data.status();
  }
  // Cropping YUV planes by offsetting the origins of each plane.
  // TODO(b/152629712): Investigate the impact of color shifting caused by the
  // bounding box with odd X or Y starting positions.
  const size_t plane_y_offset = input_data->y_row_stride * y0 + x0;
  const size_t plane_uv_offset = input_data->uv_row_stride * (y0 / 2) +
                                 input_data->uv_pixel_stride * (x0 / 2);
  Buffer::DataPlane adjusted_plane_y = {input_data->y_buffer + plane_y_offset,
                                        input_data->y_row_stride,
                                        /*pixel_stride_bytes=*/1};
  Buffer::DataPlane adjusted_plane_u = {input_data->u_buffer + plane_uv_offset,
                                        input_data->uv_row_stride,
                                        input_data->uv_pixel_stride};
  Buffer::DataPlane adjusted_plane_v = {input_data->v_buffer + plane_uv_offset,
                                        input_data->uv_row_stride,
                                        input_data->uv_pixel_stride};

  // Uniform resize is achieved by adjusting the resize dimension to fit the
  // output_buffer and respect the input aspect ratio at the same time. For
  // YUV formats, we need access to the actual output dimension to get the
  // correct address of each plane. For this, we are not calling ResizeNv or
  // ResizeYv but the libyuv scale methods directly.
  std::vector<size_t> adjusted_dimension =
      GetScaledDimension(input_dimension, output_buffer.GetDimension());

  absl::StatusOr<YuvData> output_data = GetYuvDataFromBuffer(output_buffer);

  if (!output_data.ok()) {
    return output_data.status();
  }

  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kNV12: {
      int ret = libyuv::NV12Scale(
          adjusted_plane_y.data, adjusted_plane_y.row_stride_bytes,
          adjusted_plane_u.data, adjusted_plane_u.row_stride_bytes,
          input_dimension[0], input_dimension[1],
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride, adjusted_dimension[0],
          adjusted_dimension[1], libyuv::FilterMode::kFilterBilinear);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12Scale operation failed.");
      }
      return absl::OkStatus();
    }
    case BufferFormat::kNV21: {
      int ret = libyuv::NV12Scale(
          adjusted_plane_y.data, adjusted_plane_y.row_stride_bytes,
          adjusted_plane_v.data, adjusted_plane_v.row_stride_bytes,
          input_dimension[0], input_dimension[1],
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, adjusted_dimension[0],
          adjusted_dimension[1], libyuv::FilterMode::kFilterBilinear);
      if (ret != 0) {
        return absl::UnknownError("Libyuv NV12Scale operation failed.");
      }
      return absl::OkStatus();
    }
    case BufferFormat::kYV12:
    case BufferFormat::kYV21: {
      int ret = libyuv::I420Scale(
          adjusted_plane_y.data, adjusted_plane_y.row_stride_bytes,
          adjusted_plane_u.data, adjusted_plane_u.row_stride_bytes,
          adjusted_plane_v.data, adjusted_plane_v.row_stride_bytes,
          input_dimension[0], input_dimension[1],
          const_cast<unsigned char*>(output_data->y_buffer),
          output_data->y_row_stride,
          const_cast<unsigned char*>(output_data->u_buffer),
          output_data->uv_row_stride,
          const_cast<unsigned char*>(output_data->v_buffer),
          output_data->uv_row_stride, adjusted_dimension[0],
          adjusted_dimension[1], libyuv::FilterMode::kFilterBilinear);
      if (ret != 0) {
        return absl::UnknownError("Libyuv I420Scale operation failed.");
      }
      return absl::OkStatus();
    }
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
  return absl::OkStatus();
}

absl::Status LibyuvBufferUtils::Crop(const Buffer& buffer, int x0, int y0,
                                     int x1, int y1, Buffer& output_buffer) {
  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kRGBA:
    case BufferFormat::kRGB:
    case BufferFormat::kGrayScale:
      return CropResize(buffer, x0, y0, x1, y1, output_buffer);
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return CropResizeYuv(buffer, x0, y0, x1, y1, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

absl::Status LibyuvBufferUtils::Resize(const Buffer& buffer,
                                       Buffer& output_buffer) {
  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return ResizeYv(buffer, output_buffer);
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
      return ResizeNv(buffer, output_buffer);
    case BufferFormat::kRGB:
      return ResizeRgb(buffer, output_buffer);
    case BufferFormat::kRGBA:
      return ResizeRgba(buffer, output_buffer);
    case BufferFormat::kGrayScale:
      return ResizeGray(buffer, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

absl::Status LibyuvBufferUtils::Rotate(const Buffer& buffer, int angle_deg,
                                       Buffer& output_buffer) {
  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kGrayScale:
      return RotateGray(buffer, angle_deg, output_buffer);
    case BufferFormat::kRGBA:
      return RotateRgba(buffer, angle_deg, output_buffer);
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
      return RotateNv(buffer, angle_deg, output_buffer);
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return RotateYv(buffer, angle_deg, output_buffer);
    case BufferFormat::kRGB:
      return RotateRgb(buffer, angle_deg, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

absl::Status LibyuvBufferUtils::FlipHorizontally(const Buffer& buffer,
                                                 Buffer& output_buffer) {
  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kRGBA:
      return FlipHorizontallyRgba(buffer, output_buffer);
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return FlipHorizontallyYv(buffer, output_buffer);
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
      return FlipHorizontallyNv(buffer, output_buffer);
    case BufferFormat::kRGB:
      return FlipHorizontallyRgb(buffer, output_buffer);
    case BufferFormat::kGrayScale:
      return FlipHorizontallyPlane(buffer, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

absl::Status LibyuvBufferUtils::FlipVertically(const Buffer& buffer,
                                               Buffer& output_buffer) {
  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kRGBA:
    case BufferFormat::kRGB:
    case BufferFormat::kGrayScale:
      return FlipPlaneVertically(buffer, output_buffer);
    case BufferFormat::kNV12:
    case BufferFormat::kNV21:
      return FlipVerticallyNv(buffer, output_buffer);
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return FlipVerticallyYv(buffer, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

absl::Status LibyuvBufferUtils::Convert(const Buffer& buffer,
                                        Buffer& output_buffer) {
  switch (buffer.GetBufferFormat()) {
    case BufferFormat::kNV12:
      return ConvertFromNv12(buffer, output_buffer);
    case BufferFormat::kNV21:
      return ConvertFromNv21(buffer, output_buffer);
    case BufferFormat::kYV12:
    case BufferFormat::kYV21:
      return ConvertFromYv(buffer, output_buffer);
    case BufferFormat::kRGB:
      return ConvertFromRgb(buffer, output_buffer);
    case BufferFormat::kRGBA:
      return ConvertFromRgba(buffer, output_buffer);
    default:
      return absl::InternalError(absl::StrFormat(
          "Format %s is not supported.", ToString(buffer.GetBufferFormat())));
  }
}

}  // namespace band