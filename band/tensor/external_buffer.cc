#include "band/tensor/external_buffer.h"

#include "band/logger.h"
#include "external_buffer.h"

namespace band {
std::shared_ptr<ExternalBuffer> ExternalBuffer::CreateFromBuffer(
    const char* data, size_t width, size_t height, FormatType format_type) {
  if (format_type <= FormatType::BGRA) {
    return std::make_shared<ExternalBuffer>(
        std::vector<int>{static_cast<int>(width), static_cast<int>(height)},
        std::vector<DataPlane>{{data, width * GetPixelStrideBytes(format_type),
                                GetPixelStrideBytes(format_type)}},
        format_type);
  }

  switch (format_type) {
    case FormatType::NV21: {
      const int row_stride_uv = (width % 2 == 1) ? (width + 1) / 2 * 2 : width;
      return CreateFromYUVPlanes(data,                       // y
                                 data + width * height + 1,  // u
                                 data + width * height,      // v
                                 width, height, width, row_stride_uv, 2,
                                 format_type);
    }
    case FormatType::NV12: {
      const int row_stride_uv = (width % 2 == 1) ? (width + 1) / 2 * 2 : width;
      return CreateFromYUVPlanes(data,                       // y
                                 data + width * height,      // u
                                 data + width * height + 1,  // v
                                 width, height, width, row_stride_uv, 2,
                                 format_type);
    }
    case FormatType::YV21: {
      std::vector<int> uv_dims = GetUvDims(
          std::vector<int>{static_cast<int>(width), static_cast<int>(height)},
          format_type);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height,                            // u
          data + width * height + uv_dims[0] * uv_dims[1],  // v
          width, height, width, uv_dims[0], 1, format_type);
    }
    case FormatType::YV12: {
      std::vector<int> uv_dims = GetUvDims(
          std::vector<int>{static_cast<int>(width), static_cast<int>(height)},
          format_type);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height + uv_dims[0] * uv_dims[1],  // u
          data + width * height,                            // v
          width, height, width, uv_dims[0], 1, format_type);
    }
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported format type : %s",
                    GetName(format_type));
      return nullptr;
  }
}

std::shared_ptr<ExternalBuffer> ExternalBuffer::CreateFromYUVPlanes(
    const char* y_data, const char* u_data, const char* v_data, size_t width,
    size_t height, size_t row_stride_y, size_t row_stride_uv,
    size_t pixel_stride_uv, FormatType format_type) {
  std::vector<DataPlane> data_planes;
  if (format_type == FormatType::NV21 || format_type == FormatType::YV12) {
    data_planes = {{y_data, row_stride_y, 1},
                   {v_data, row_stride_uv, pixel_stride_uv},
                   {u_data, row_stride_uv, pixel_stride_uv}};
  } else if (format_type == FormatType::NV12 ||
             format_type == FormatType::YV21) {
    data_planes = {{y_data, row_stride_y, 1},
                   {u_data, row_stride_uv, pixel_stride_uv},
                   {v_data, row_stride_uv, pixel_stride_uv}};
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported YUV format type : %s",
                  GetName(format_type));
    return nullptr;
  }

  return std::make_shared<ExternalBuffer>(
      std::vector<int>{static_cast<int>(width), static_cast<int>(height)},
      data_planes, format_type);
}

ExternalBuffer::ExternalBuffer(const std::vector<int>& dims,
                               const std::vector<DataPlane>& data_planes,
                               FormatType format_type)
    : dims_(dims), data_planes_(data_planes), format_type_(format_type) {}

size_t ExternalBuffer::GetPixelStrideBytes(FormatType format_type) {
  switch (format_type) {
    case FormatType::GrayScale:
      return 1;
    case FormatType::RGB:
    case FormatType::BGR:
      return 3;
    case FormatType::RGBA:
    case FormatType::BGRA:
      return 4;
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR,
                    "Given format type requires external input to guess the "
                    "pixel stride : %s",
                    GetName(format_type));
      return 0;
  }
}
std::vector<int> ExternalBuffer::GetUvDims(const std::vector<int>& dims,
                                           FormatType format_type) {
  if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0) {
    std::string dims_str;
    for (const auto& dim : dims) {
      dims_str += std::to_string(dim) + " ";
    }
    BAND_LOG_PROD(BAND_LOG_ERROR, "Given dims is not valid for UV plane : %s",
                  dims_str.c_str());
    return std::vector<int>();
  }

  switch (format_type) {
    case FormatType::NV21:
    case FormatType::NV12:
    case FormatType::YV21:
    case FormatType::YV12:
      return {(dims[0] + 1) / 2, (dims[1] + 1) / 2};
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported format type : %s",
                    GetName(format_type));
      return std::vector<int>();
  }
}

size_t ExternalBuffer::GetSize(const std::vector<int>& dims) {
  size_t size = 1;
  for (const auto& dim : dims) {
    if (dim <= 0) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Given dims is not valid : %d", dim);
      return 0;
    }
    size *= dim;
  }
  return size;
}
}  // namespace band