#include "band/buffer/buffer.h"

#include "band/logger.h"
#include "buffer.h"

namespace band {

Buffer::~Buffer() {
  if (owns_data_) {
    // only the first data plane owns the data
    delete[] data_planes_[0].data;
  }
}

std::shared_ptr<Buffer> Buffer::CreateFromPlanes(
    const std::vector<DataPlane>& data_planes, const std::vector<size_t>& dims,
    BufferFormat buffer_format, BufferOrientation orientation) {
  return std::shared_ptr<Buffer>(
      new Buffer(dims, data_planes, buffer_format, orientation));
}

std::shared_ptr<Buffer> Buffer::CreateFromRaw(const unsigned char* data,
                                              size_t width, size_t height,
                                              BufferFormat buffer_format,
                                              BufferOrientation orientation,
                                              bool owns_data) {
  if (buffer_format <= BufferFormat::kBandRGBA) {
    return std::shared_ptr<Buffer>(new Buffer(
        std::vector<size_t>{width, height},
        std::vector<DataPlane>{{data,
                                width * GetPixelStrideBytes(buffer_format),
                                GetPixelStrideBytes(buffer_format)}},
        buffer_format, orientation, owns_data));
  }

  switch (buffer_format) {
    case BufferFormat::kBandNV21: {
      const int row_stride_uv = (width % 2 == 1) ? (width + 1) / 2 * 2 : width;
      return CreateFromYUVPlanes(data,                       // y
                                 data + width * height + 1,  // u
                                 data + width * height,      // v
                                 width, height, width, row_stride_uv, 2,
                                 buffer_format, orientation, owns_data);
    }
    case BufferFormat::kBandNV12: {
      const int row_stride_uv = (width % 2 == 1) ? (width + 1) / 2 * 2 : width;
      return CreateFromYUVPlanes(data,                       // y
                                 data + width * height,      // u
                                 data + width * height + 1,  // v
                                 width, height, width, row_stride_uv, 2,
                                 buffer_format, orientation, owns_data);
    }
    case BufferFormat::kBandYV21: {
      std::vector<size_t> uv_dims =
          GetUvDims(std::vector<size_t>{width, height}, buffer_format);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height,                            // u
          data + width * height + uv_dims[0] * uv_dims[1],  // v
          width, height, width, uv_dims[0], 1, buffer_format, orientation,
          owns_data);
    }
    case BufferFormat::kBandYV12: {
      std::vector<size_t> uv_dims =
          GetUvDims(std::vector<size_t>{width, height}, buffer_format);
      return CreateFromYUVPlanes(
          data,                                             // y
          data + width * height + uv_dims[0] * uv_dims[1],  // u
          data + width * height,                            // v
          width, height, width, uv_dims[0], 1, buffer_format, orientation,
          owns_data);
    }
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported format type : %s",
                    ToString(buffer_format));
      return nullptr;
  }
}

std::shared_ptr<Buffer> Buffer::CreateFromYUVPlanes(
    const unsigned char* y_data, const unsigned char* u_data,
    const unsigned char* v_data, size_t width, size_t height,
    size_t row_stride_y, size_t row_stride_uv, size_t pixel_stride_uv,
    BufferFormat buffer_format, BufferOrientation orientation, bool owns_data) {
  std::vector<DataPlane> data_planes;
  if (buffer_format == BufferFormat::kBandNV21 ||
      buffer_format == BufferFormat::kBandYV12) {
    data_planes = {{y_data, row_stride_y, 1},
                   {v_data, row_stride_uv, pixel_stride_uv},
                   {u_data, row_stride_uv, pixel_stride_uv}};
  } else if (buffer_format == BufferFormat::kBandNV12 ||
             buffer_format == BufferFormat::kBandYV21) {
    data_planes = {{y_data, row_stride_y, 1},
                   {u_data, row_stride_uv, pixel_stride_uv},
                   {v_data, row_stride_uv, pixel_stride_uv}};
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported YUV format type : %s",
                  ToString(buffer_format));
    return nullptr;
  }

  return std::shared_ptr<Buffer>(new Buffer(std::vector<size_t>{width, height},
                                            data_planes, buffer_format,
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
    // ignore the dimension with size 1
    if (tensor->GetDims()[i] == 1) {
      continue;
    }

    dims.push_back(tensor->GetDims()[i]);
    if (dims.back() <= 0) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Given tensor has invalid dimension : %d",
                    dims.back());
      return nullptr;
    }
  }

  std::vector<DataPlane> data_planes;
  bool is_rgb = dims.size() == 3 && dims[2] == 3;

  if (is_rgb) {
    // assume the tensor is in NHWC format
    dims = {dims[1], dims[0]};
    data_planes.push_back(
        DataPlane{reinterpret_cast<const unsigned char*>(tensor->GetData()),
                  dims[0] * 3, 3});  // RGB
    return std::shared_ptr<Buffer>(new Buffer(
        dims, data_planes, BufferFormat::kBandRGB, BufferOrientation::kBandTopLeft));
  } else {
    // flatten the tensor into single-row data plane
    data_planes.push_back(
        DataPlane{reinterpret_cast<const unsigned char*>(tensor->GetData()),
                  tensor->GetBytes(), tensor->GetPixelBytes()});

    return std::shared_ptr<Buffer>(new Buffer(
        dims, data_planes, tensor->GetType(), BufferOrientation::kBandTopLeft));
  }
}

std::shared_ptr<Buffer> Buffer::CreateEmpty(size_t width, size_t height,
                                            BufferFormat buffer_format,
                                            BufferOrientation orientation) {
  size_t total_bytes = GetSize({width, height});

  switch (buffer_format) {
    case BufferFormat::kBandGrayScale:
    case BufferFormat::kBandRGB:
    case BufferFormat::kBandRGBA: {
      // pixel stride bytes
      total_bytes *= GetPixelStrideBytes(buffer_format);
      break;
    }

    case BufferFormat::kBandNV21:
    case BufferFormat::kBandNV12:
    case BufferFormat::kBandYV21:
    case BufferFormat::kBandYV12: {
      // uv plane has 2 bytes per pixel
      total_bytes += GetSize(GetUvDims({width, height}, buffer_format)) * 2;
      break;
    }

    case BufferFormat::kBandRaw: {
      BAND_LOG_PROD(BAND_LOG_ERROR,
                    "Raw format type requires external input to create "
                    "empty buffer");
      return nullptr;
    }
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported format type : %s",
                    ToString(buffer_format));
      return nullptr;
  }

  return CreateFromRaw(new unsigned char[total_bytes], width, height,
                       buffer_format, orientation, true);
}

Buffer::Buffer(std::vector<size_t> dimension,
               std::vector<DataPlane> data_planes, BufferFormat buffer_format,
               BufferOrientation orientation, bool owns_data)
    : dimension_(dimension),
      data_planes_(data_planes),
      buffer_format_(buffer_format),
      orientation_(orientation),
      owns_data_(owns_data) {}

Buffer::Buffer(std::vector<size_t> dimension,
               std::vector<DataPlane> data_planes, DataType data_type,
               BufferOrientation orientation, bool owns_data)
    : Buffer(dimension, data_planes, BufferFormat::kBandRaw, orientation,
             owns_data) {
  data_type_ = data_type;
}

size_t Buffer::GetPixelStrideBytes(BufferFormat buffer_format) {
  switch (buffer_format) {
    case BufferFormat::kBandGrayScale:
      return 1;
    case BufferFormat::kBandRGB:
      return 3;
    case BufferFormat::kBandRGBA:
      return 4;
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR,
                    "Given format type requires external input to guess the "
                    "pixel stride : %s",
                    ToString(buffer_format));
      return 0;
  }
}

std::vector<size_t> Buffer::GetUvDims(const std::vector<size_t>& dims,
                                      BufferFormat buffer_format) {
  if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0) {
    std::string dims_str;
    for (const auto& dim : dims) {
      dims_str += std::to_string(dim) + " ";
    }
    BAND_LOG_PROD(BAND_LOG_ERROR, "Given dims is not valid for UV plane : %s",
                  dims_str.c_str());
    return std::vector<size_t>();
  }

  switch (buffer_format) {
    case BufferFormat::kBandNV21:
    case BufferFormat::kBandNV12:
    case BufferFormat::kBandYV21:
    case BufferFormat::kBandYV12:
      return {(dims[0] + 1) / 2, (dims[1] + 1) / 2};
    default:
      BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported format type : %s",
                    ToString(buffer_format));
      return std::vector<size_t>();
  }
}

size_t Buffer::GetBufferByteSize(const std::vector<size_t>& dims,
                                 BufferFormat buffer_format) {
  switch (buffer_format) {
    case BufferFormat::kBandNV21:
    case BufferFormat::kBandNV12:
    case BufferFormat::kBandYV21:
    case BufferFormat::kBandYV12: {
      std::vector<size_t> uv_dims = GetUvDims(dims, buffer_format);
      if (uv_dims.empty()) {
        return 0;
      }
      return GetSize(dims) +        // y plane
             GetSize(uv_dims) * 2;  // uv plane has 2 bytes per pixel
    }
    default:
      return GetSize(dims) * GetPixelStrideBytes(buffer_format);
  }
}

std::vector<size_t> Buffer::GetCropDimension(size_t x0, size_t x1, size_t y0,
                                             size_t y1) {
  return {x1 - x0 + 1, y1 - y0 + 1};
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
  if (buffer_format_ == BufferFormat::kBandRaw) {
    // custom format type has only one data plane
    return data_planes_[0].pixel_stride_bytes;
  } else {
    return GetPixelStrideBytes(buffer_format_);
  }
}

size_t Buffer::GetBytes() const { return GetPixelBytes() * GetNumElements(); }

DataType Buffer::GetDataType() const { return data_type_; }

BufferFormat Buffer::GetBufferFormat() const { return buffer_format_; }

BufferOrientation Buffer::GetOrientation() const { return orientation_; }

bool Buffer::IsBufferFormatCompatible(const Buffer& rhs) const {
  switch (buffer_format_) {
    case BufferFormat::kBandRGB:
    case BufferFormat::kBandRGBA:
      return rhs.buffer_format_ == BufferFormat::kBandRGB ||
             rhs.buffer_format_ == BufferFormat::kBandRGBA;
    case BufferFormat::kBandNV21:
    case BufferFormat::kBandNV12:
    case BufferFormat::kBandYV21:
    case BufferFormat::kBandYV12:
      return rhs.buffer_format_ == BufferFormat::kBandNV21 ||
             rhs.buffer_format_ == BufferFormat::kBandNV12 ||
             rhs.buffer_format_ == BufferFormat::kBandYV21 ||
             rhs.buffer_format_ == BufferFormat::kBandYV12;
    default:
      return buffer_format_ == rhs.buffer_format_;
  }
}
}  // namespace band