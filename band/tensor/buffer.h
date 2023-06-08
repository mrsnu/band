#ifndef BAND_TENSOR_BUFFER_H_
#define BAND_TENSOR_BUFFER_H_

#include <memory>

#include "band/common.h"
#include "band/interface/tensor.h"

namespace band {
namespace tensor {

// Buffer content orientation follows EXIF specification. The name of
// each enum value defines the position of the 0th row and the 0th column of
// the image content. See http://jpegclub.org/exif_orientation.html for
// details.
enum class Orientation {
  TopLeft = 1,
  TopRight = 2,
  BottomRight = 3,
  BottomLeft = 4,
  LeftTop = 5,
  RightTop = 6,
  RightBottom = 7,
  LeftBottom = 8
};

// buffer with multiple data planes. Each data plane has its own demension.
class Buffer {
 public:
  struct DataPlane {
    const unsigned char* data;
    size_t row_stride_bytes = 1;
    size_t pixel_stride_bytes = 1;
  };
  ~Buffer();

  static std::shared_ptr<Buffer> CreateFromPlanes(
      const std::vector<DataPlane>& data_planes,
      const std::vector<size_t>& dims, FormatType format_type,
      Orientation orientation = Orientation::TopLeft);
  static std::shared_ptr<Buffer> CreateFromRaw(
      const unsigned char* data, size_t width, size_t height,
      FormatType format_type, Orientation orientation = Orientation::TopLeft,
      bool owns_data = false);
  static std::shared_ptr<Buffer> CreateFromYUVPlanes(
      const unsigned char* y_data, const unsigned char* u_data,
      const unsigned char* v_data, size_t width, size_t height,
      size_t row_stride_y, size_t row_stride_uv, size_t pixel_stride_uv,
      FormatType format_type, Orientation orientation = Orientation::TopLeft,
      bool owns_data = false);
  static std::shared_ptr<Buffer> CreateFromTensor(
      const interface::ITensor* tensor);
  static std::shared_ptr<Buffer> CreateEmpty(
      size_t width, size_t height, FormatType format_type,
      Orientation orientation = Orientation::TopLeft);

  const std::vector<size_t>& GetDimension() const;
  const DataPlane& operator[](size_t index) const;
  DataPlane& operator[](size_t index);
  size_t GetNumPlanes() const;
  size_t GetNumElements() const;
  size_t GetPixelBytes() const;
  size_t GetBytes() const;
  FormatType GetFormatType() const;
  Orientation GetOrientation() const;

  bool IsFormatTypeCompatible(const Buffer& rhs) const;

  static std::vector<size_t> GetUvDims(const std::vector<size_t>& dims,
                                       FormatType format_type);
  static size_t GetBufferByteSize(const std::vector<size_t>& dims,
                                  FormatType format_type);

 private:
  Buffer(std::vector<size_t> dimension, std::vector<DataPlane> data_planes,
         FormatType format_type, Orientation orientation,
         bool owns_data = false);

  static size_t GetPixelStrideBytes(FormatType format_type);
  static size_t GetSize(const std::vector<size_t>& dims);

  bool owns_data_;
  const std::vector<size_t> dimension_;
  std::vector<DataPlane> data_planes_;
  FormatType format_type_;
  Orientation orientation_;
};
}  // namespace tensor
}  // namespace band

#endif