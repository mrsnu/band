/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    unsigned char* GetMutableData() { return const_cast<unsigned char*>(data); }
    const unsigned char* data;
    // row_stride_bytes is the number of bytes between two consecutive rows.
    // must equal to width * pixel_stride_bytes for interleaved data.
    size_t row_stride_bytes = 1;
    size_t pixel_stride_bytes = 1;
  };
  ~Buffer();

  static Buffer* CreateFromPlanes(
      const std::vector<DataPlane>& data_planes,
      const std::vector<size_t>& dims, BufferFormat buffer_format,
      BufferOrientation orientation = BufferOrientation::kTopLeft);

  static Buffer* CreateFromRaw(
      const unsigned char* data, size_t width, size_t height,
      BufferFormat buffer_format, DataType data_type = DataType::kUInt8,
      BufferOrientation orientation = BufferOrientation::kTopLeft,
      bool owns_data = false);

  static Buffer* CreateFromYUVPlanes(
      const unsigned char* y_data, const unsigned char* u_data,
      const unsigned char* v_data, size_t width, size_t height,
      size_t row_stride_y, size_t row_stride_uv, size_t pixel_stride_uv,
      BufferFormat buffer_format,
      BufferOrientation orientation = BufferOrientation::kTopLeft,
      bool owns_data = false);

  static Buffer* CreateFromTensor(const interface::ITensor* tensor);

  static Buffer* CreateEmpty(
      size_t width, size_t height, BufferFormat buffer_format,
      DataType data_type,
      BufferOrientation orientation = BufferOrientation::kTopLeft);

  const std::vector<size_t>& GetDimension() const;
  const DataPlane& operator[](size_t index) const;
  DataPlane& operator[](size_t index);
  size_t GetNumPlanes() const;
  size_t GetNumElements() const;
  size_t GetPixelBytes() const;
  size_t GetNumPixelElements() const;
  size_t GetBytes() const;
  DataType GetDataType() const;
  BufferFormat GetBufferFormat() const;
  BufferOrientation GetOrientation() const;

  absl::Status CopyFrom(const Buffer& rhs);

  bool IsBufferFormatCompatible(const Buffer& rhs) const;

  static std::vector<size_t> GetUvDims(const std::vector<size_t>& dims,
                                       BufferFormat buffer_format);
  static size_t GetBufferByteSize(const std::vector<size_t>& dims,
                                  BufferFormat buffer_format);
  static bool IsYUV(BufferFormat buffer_format);

  static std::vector<size_t> GetCropDimension(size_t x0, size_t x1, size_t y0,
                                              size_t y1);

 private:
  Buffer(std::vector<size_t> dimension, std::vector<DataPlane> data_planes,
         BufferFormat buffer_format, DataType data_type,
         BufferOrientation orientation, bool owns_data = false);
  Buffer(std::vector<size_t> dimension, std::vector<DataPlane> data_planes,
         DataType data_type, BufferOrientation orientation,
         bool owns_data = false);

  static size_t GetNumPixelElements(BufferFormat buffer_format);
  static size_t GetSize(const std::vector<size_t>& dims);

  bool owns_data_;
  const std::vector<size_t> dimension_;
  std::vector<DataPlane> data_planes_;
  BufferFormat buffer_format_;
  DataType data_type_ = DataType::kUInt8;
  BufferOrientation orientation_;
};

std::ostream& operator<<(std::ostream& os, const Buffer& buffer);

}  // namespace band

#endif