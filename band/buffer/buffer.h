#ifndef BAND_BUFFER_BUFFER_H_
#define BAND_BUFFER_BUFFER_H_

#include <memory>

#include "band/common.h"
#include "band/interface/tensor.h"

namespace band {

// External buffer with multiple data planes. Each data plane has its own
// demension. The main purpose of this class is to provide a unified interface
// for external user to pass in buffer data to the engine. The engine will then
// convert the buffer to the internal tensor format.
class Buffer {
 public:
  struct DataPlane {
    const unsigned char* data;
    // row_stride_bytes is the number of bytes between two consecutive rows.
    size_t row_stride_bytes = 1;
    size_t pixel_stride_bytes = 1;
  };
  ~Buffer();

  static std::shared_ptr<Buffer> CreateFromPlanes(
      const std::vector<DataPlane>& data_planes,
      const std::vector<size_t>& dims, BufferFormat buffer_format,
      BufferOrientation orientation = BufferOrientation::kBandTopLeft);

  static std::shared_ptr<Buffer> CreateFromRaw(
      const unsigned char* data, size_t width, size_t height,
      BufferFormat buffer_format,
      BufferOrientation orientation = BufferOrientation::kBandTopLeft,
      bool owns_data = false);

  static std::shared_ptr<Buffer> CreateFromYUVPlanes(
      const unsigned char* y_data, const unsigned char* u_data,
      const unsigned char* v_data, size_t width, size_t height,
      size_t row_stride_y, size_t row_stride_uv, size_t pixel_stride_uv,
      BufferFormat buffer_format,
      BufferOrientation orientation = BufferOrientation::kBandTopLeft,
      bool owns_data = false);

  static std::shared_ptr<Buffer> CreateFromTensor(
      const interface::ITensor* tensor);

  static std::shared_ptr<Buffer> CreateEmpty(
      size_t width, size_t height, BufferFormat buffer_format,
      BufferOrientation orientation = BufferOrientation::kBandTopLeft);

  const std::vector<size_t>& GetDimension() const;
  const DataPlane& operator[](size_t index) const;
  DataPlane& operator[](size_t index);
  size_t GetNumPlanes() const;
  size_t GetNumElements() const;
  size_t GetPixelBytes() const;
  size_t GetBytes() const;
  DataType GetDataType() const;
  BufferFormat GetBufferFormat() const;
  BufferOrientation GetOrientation() const;

  bool IsBufferFormatCompatible(const Buffer& rhs) const;

  static std::vector<size_t> GetUvDims(const std::vector<size_t>& dims,
                                       BufferFormat buffer_format);
  static size_t GetBufferByteSize(const std::vector<size_t>& dims,
                                  BufferFormat buffer_format);

  static std::vector<size_t> GetCropDimension(size_t x0, size_t x1, size_t y0,
                                              size_t y1);

 private:
  Buffer(std::vector<size_t> dimension, std::vector<DataPlane> data_planes,
         BufferFormat buffer_format, BufferOrientation orientation,
         bool owns_data = false);
  Buffer(std::vector<size_t> dimension, std::vector<DataPlane> data_planes,
         DataType data_type, BufferOrientation orientation,
         bool owns_data = false);

  static size_t GetPixelStrideBytes(BufferFormat buffer_format);
  static size_t GetSize(const std::vector<size_t>& dims);

  bool owns_data_;
  const std::vector<size_t> dimension_;
  std::vector<DataPlane> data_planes_;
  BufferFormat buffer_format_;
  DataType data_type_ = DataType::kBandUInt8;
  BufferOrientation orientation_;
};

}  // namespace band

#endif