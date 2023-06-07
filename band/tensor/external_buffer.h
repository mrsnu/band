#ifndef BAND_TENSOR_EXTERNAL_BUFFER_H_
#define BAND_TENSOR_EXTERNAL_BUFFER_H_

#include <memory>

#include "band/common.h"

namespace band {
class ExternalBuffer {
 public:
  static std::shared_ptr<ExternalBuffer> CreateFromBuffer(
      const char* data, size_t width, size_t height, FormatType format_type);
  static std::shared_ptr<ExternalBuffer> CreateFromYUVPlanes(
      const char* y_data, const char* u_data, const char* v_data, size_t width,
      size_t height, size_t row_stride_y, size_t row_stride_uv,
      size_t pixel_stride_uv, FormatType format_type);

 private:
  struct DataPlane {
    // owned by the caller
    const char* data;
    size_t row_stride_bytes = 1;
    size_t pixel_stride_bytes = 1;
  };

  ExternalBuffer(const std::vector<int>& dims,
                 const std::vector<DataPlane>& data_planes,
                 FormatType format_type);
  ExternalBuffer() = default;
  ~ExternalBuffer() = default;

  static size_t GetPixelStrideBytes(FormatType format_type);
  static std::vector<int> GetUvDims(const std::vector<int>& dims,
                                    FormatType format_type);
  static size_t GetSize(const std::vector<int>& dims);

  std::vector<int> dims_;
  std::vector<DataPlane> data_planes_;
  FormatType format_type_;
};
}  // namespace band

#endif