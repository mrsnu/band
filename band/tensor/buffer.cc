#include "band/tensor/buffer.h"

#include "band/logger.h"
#include "buffer.h"

namespace band {
namespace tensor {

Buffer::~Buffer() {
  if (owns_data_) {
    // only the first data plane owns the data
    delete[] data_planes_[0].data;
  }
}

std::shared_ptr<Buffer> Buffer::CreateFromPlanes(
    const std::vector<DataPlane>& data_planes, const std::vector<size_t>& dims,
    FormatType format_type, Orientation orientation) {
  return std::shared_ptr<Buffer>(
      new Buffer(dims, data_planes, format_type, orientation));
}

std::shared_ptr<Buffer> Buffer::CreateFromRaw(const unsigned char* data,
                                              size_t width, size_t height,
                                              FormatType format_type,
                                              Orientation orientation,
                                              bool owns_data) {
  if (format_type <= FormatType::RGBA) {
    return std::shared_ptr<Buffer>(new Buffer(
        std::vector<size_t>{width, height},
        std::vector<DataPlane>{{data, width * GetPixelStrideBytes(format_type),
                                GetPixelStrideBytes(format_type)}},
        format_type, orientation, owns_data));
  }

  switch (format_type) {
    case FormatType::NV21: {
      const int row_stride_uv = (width % 2 == 1) ? (width + 1) / 2 * 2 : width;
      return CreateFromYUVPlanes(data,                       // y
                                 data + width * height + 1,  // u
                                 data + width * height,      // v
                                 width, height, width, row_stride_uv, 2,
                                 format_type, orientation, owns_data);
    }
    case FormatType::NV12: {
      const int row_stride_uv = (width % 2 == 1) ? (width + 1) / 2 * 2 : width;
      return CreateFromYUVPlanes(data,                       // y
                                 data + width * height,      // u
                                 data + width * height + 1,  // v
                                 width, height, width, row_stride_uv, 2,
                                 format_type, orientation, owns_data);
    }
    case FormatType::YV21: {
      std::vector<size_t> uv_dims =
          GetUvDims(std::vector<size_t>{width, height}, format_type);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height,                            // u
          data + width * height + uv_dims[0] * uv_dims[1],  // v
          width, height, width, uv_dims[0], 1, format_type, orientation,
          owns_data);
    }
    case FormatType::YV12: {
      std::vector<size_t> uv_dims =
          GetUvDims(std::vector<size_t>{width, height}, format_type);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height + uv_dims[0] * uv_dims[1],  // u
          data + width * height,                            // v
          width, height, width, uv_dims[0], 1, format_type, orientation,
          owns_data);
    }
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported format type : %s",
                    GetName(format_type));
      return nullptr;
  }
}

std::shared_ptr<Buffer> Buffer::CreateFromYUVPlanes(
    const unsigned char* y_data, const unsigned char* u_data,
    const unsigned char* v_data, size_t width, size_t height,
    size_t row_stride_y, size_t row_stride_uv, size_t pixel_stride_uv,
    FormatType format_type, Orientation orientation, bool owns_data) {
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

  return std::shared_ptr<Buffer>(new Buffer(std::vector<size_t>{width, height},
                                            data_planes, format_type,
                                            orientation, owns_data));
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
  data_planes.push_back(DataPlane{
      reinterpret_cast<const unsigned char*>(tensor->GetData()),
      tensor->GetDims()[0] * tensor->GetPixelBytes(), tensor->GetPixelBytes()});

  return std::shared_ptr<Buffer>(
      new Buffer(dims, data_planes, FormatType::Custom, Orientation::TopLeft));
}

std::shared_ptr<Buffer> Buffer::CreateEmpty(size_t width, size_t height,
                                            FormatType format_type,
                                            Orientation orientation) {
  size_t total_bytes = GetSize({width, height});

  switch (format_type) {
    case FormatType::GrayScale:
    case FormatType::RGB:
    case FormatType::RGBA: {
      // pixel stride bytes
      total_bytes *= GetPixelStrideBytes(format_type);
      break;
    }

    case FormatType::NV21:
    case FormatType::NV12:
    case FormatType::YV21:
    case FormatType::YV12: {
      // uv plane has 2 bytes per pixel
      total_bytes += GetSize(GetUvDims({width, height}, format_type)) * 2;
      break;
    }

    case FormatType::Custom: {
      BAND_LOG_PROD(BAND_LOG_ERROR,
                    "Custom format type requires external input to create "
                    "empty buffer");
      return nullptr;
    }
  }

  return CreateFromRaw(new unsigned char[total_bytes], width, height,
                       format_type, orientation, true);
}

Buffer::Buffer(std::vector<size_t> dimension,
               std::vector<DataPlane> data_planes, FormatType format_type,
               Orientation orientation, bool owns_data)
    : dimension_(dimension),
      data_planes_(data_planes),
      format_type_(format_type),
      orientation_(orientation),
      owns_data_(owns_data) {}

size_t Buffer::GetPixelStrideBytes(FormatType format_type) {
  switch (format_type) {
    case FormatType::GrayScale:
      return 1;
    case FormatType::RGB:
      return 3;
    case FormatType::RGBA:
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

size_t Buffer::GetBufferByteSize(const std::vector<size_t>& dims,
                                 FormatType format_type) {
  switch (format_type) {
    case FormatType::NV21:
    case FormatType::NV12:
    case FormatType::YV21:
    case FormatType::YV12: {
      std::vector<size_t> uv_dims = GetUvDims(dims, format_type);
      if (uv_dims.empty()) {
        return 0;
      }
      return GetSize(dims) +        // y plane
             GetSize(uv_dims) * 2;  // uv plane has 2 bytes per pixel
    }
    default:
      return GetSize(dims) * GetPixelStrideBytes(format_type);
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

const std::vector<size_t>& Buffer::GetDimension() const { return dimension_; }

const Buffer::DataPlane& Buffer::operator[](size_t index) const {
  return data_planes_[index];
}

Buffer::DataPlane& Buffer::operator[](size_t index) {
  return data_planes_[index];
}

size_t Buffer::GetNumPlanes() const { return data_planes_.size(); }

size_t Buffer::GetNumElements() const {
  size_t num_elements = 1;
  for (auto dim : GetDimension()) {
    num_elements *= dim;
  }
  return num_elements;
}

size_t Buffer::GetPixelBytes() const {
  if (format_type_ == FormatType::Custom) {
    // custom format type has only one data plane
    return data_planes_[0].pixel_stride_bytes;
  } else {
    return GetPixelStrideBytes(format_type_);
  }
}

size_t Buffer::GetBytes() const { return GetPixelBytes() * GetNumElements(); }

FormatType Buffer::GetFormatType() const { return format_type_; }

Orientation Buffer::GetOrientation() const { return orientation_; }

bool Buffer::IsFormatTypeCompatible(const Buffer& rhs) const {
  switch (format_type_) {
    case FormatType::RGB:
    case FormatType::RGBA:
      return rhs.format_type_ == FormatType::RGB ||
             rhs.format_type_ == FormatType::RGBA;
    case FormatType::NV21:
    case FormatType::NV12:
    case FormatType::YV21:
    case FormatType::YV12:
      return rhs.format_type_ == FormatType::NV21 ||
             rhs.format_type_ == FormatType::NV12 ||
             rhs.format_type_ == FormatType::YV21 ||
             rhs.format_type_ == FormatType::YV12;
    default:
      return format_type_ == rhs.format_type_;
  }
}

}  // namespace tensor
}  // namespace band