#include "band/tensor/buffer.h"

#include "band/logger.h"
#include "buffer.h"

namespace band {
namespace tensor {
std::shared_ptr<Buffer> Buffer::CreateFromRaw(const char* data, size_t width,
                                              size_t height,
                                              FormatType format_type) {
  if (format_type <= FormatType::BGRA) {
    return std::shared_ptr<Buffer>(new Buffer(
        std::vector<size_t>{width, height},
        std::vector<DataPlane>{{data, width * GetPixelStrideBytes(format_type),
                                GetPixelStrideBytes(format_type)}},
        format_type));
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
      std::vector<size_t> uv_dims =
          GetUvDims(std::vector<size_t>{width, height}, format_type);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height,                            // u
          data + width * height + uv_dims[0] * uv_dims[1],  // v
          width, height, width, uv_dims[0], 1, format_type);
    }
    case FormatType::YV12: {
      std::vector<size_t> uv_dims =
          GetUvDims(std::vector<size_t>{width, height}, format_type);
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

std::shared_ptr<Buffer> Buffer::CreateFromYUVPlanes(
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

  return std::shared_ptr<Buffer>(
      new Buffer(std::vector<size_t>{width, height}, data_planes, format_type));
}

std::shared_ptr<Buffer> Buffer::CreateFromTensor(
    const interface::ITensor* tensor) {
  if (tensor == nullptr) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Given tensor is null");
    return nullptr;
  }

  if (tensor->GetNumDims() == 0) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Given tensor has no dimension");
    return nullptr;
  }

  std::vector<size_t> dims;
  for (size_t i = 0; i < tensor->GetNumDims(); ++i) {
    dims.push_back(tensor->GetDims()[i]);
    if (dims.back() <= 0) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Given tensor has invalid dimension : %d",
                    dims.back());
      return nullptr;
    }
  }

  std::vector<DataPlane> data_planes;
  data_planes.push_back({tensor->GetData(),
                         tensor->GetDims()[0] * tensor->GetPixelBytes(),
                         tensor->GetPixelBytes()});

  return std::shared_ptr<Buffer>(
      new Buffer(dims, data_planes, FormatType::Custom));
}

Buffer::Buffer(std::vector<size_t> dimension,
               std::vector<DataPlane> data_planes, FormatType format_type)
    : dimension_(dimension),
      data_planes_(data_planes),
      format_type_(format_type) {}

size_t Buffer::GetPixelStrideBytes(FormatType format_type) {
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

std::vector<size_t> Buffer::GetUvDims(const std::vector<size_t>& dims,
                                      FormatType format_type) {
  if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0) {
    std::string dims_str;
    for (const auto& dim : dims) {
      dims_str += std::to_string(dim) + " ";
    }
    BAND_LOG_PROD(BAND_LOG_ERROR, "Given dims is not valid for UV plane : %s",
                  dims_str.c_str());
    return std::vector<size_t>();
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
      return std::vector<size_t>();
  }
}

size_t Buffer::GetSize(const std::vector<size_t>& dims) {
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

}  // namespace tensor
}  // namespace band