#ifndef BAND_TENSOR_BUFFER_H_
#define BAND_TENSOR_BUFFER_H_

#include <memory>

#include "band/common.h"
#include "band/interface/tensor.h"

namespace band {
namespace tensor {
// buffer with multiple data planes. Each data plane has its own demension.
class Buffer {
 public:
  struct Rect {
    size_t x;
    size_t y;
    size_t width;
    size_t height;
  };
  ~Buffer() = default;

  static std::shared_ptr<Buffer> CreateFromRaw(const char* data, size_t width,
                                               size_t height,
                                               FormatType format_type);
  static std::shared_ptr<Buffer> CreateFromYUVPlanes(
      const char* y_data, const char* u_data, const char* v_data, size_t width,
      size_t height, size_t row_stride_y, size_t row_stride_uv,
      size_t pixel_stride_uv, FormatType format_type);
  static std::shared_ptr<Buffer> CreateFromTensor(
      const interface::ITensor* tensor);

 private:
  struct DataPlane {
    // owned by the caller
    const char* data;
    size_t row_stride_bytes = 1;
    size_t pixel_stride_bytes = 1;
  };

  Buffer(std::vector<size_t> dimension, std::vector<DataPlane> data_planes,
         FormatType format_type);

  static size_t GetPixelStrideBytes(FormatType format_type);
  static std::vector<size_t> GetUvDims(const std::vector<size_t>& dims,
                                       FormatType format_type);
  static size_t GetSize(const std::vector<size_t>& dims);

  const std::vector<size_t> dimension_;
  std::vector<DataPlane> data_planes_;
  FormatType format_type_;
};
}  // namespace tensor
}  // namespace band

#endif