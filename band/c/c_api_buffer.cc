// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/c/c_api_buffer.h"

#include "band/buffer/buffer.h"
#include "band/buffer/buffer_processor.h"
#include "band/buffer/common_operator.h"
#include "band/buffer/image_operator.h"
#include "band/c/c_api_internal.h"
#include "band/logger.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

using namespace band;

BandBuffer* BandBufferCreate() { return new BandBuffer(); }

void BandBufferDelete(BandBuffer* buffer) {
  if (!buffer) {
    BAND_LOG(LogSeverity::kError, "BandBuffer is null");
  }

  delete buffer;
}

BandStatus BandBufferSetFromRawData(BandBuffer* buffer, const void* data,
                                    size_t width, size_t height,
                                    BandBufferFormat format) {
  if (!buffer || !data) {
    BAND_LOG(LogSeverity::kError, "BandBuffer (%d) or data (%d) is null",
             buffer, data);
    return BandStatus::kBandErr;
  }

  buffer->impl = std::shared_ptr<band::Buffer>(
      band::Buffer::CreateFromRaw(static_cast<const unsigned char*>(data),
                                  width, height, BufferFormat(format)));
  return buffer->impl ? BandStatus::kBandOk : BandStatus::kBandErr;
}

BandStatus BandBufferSetFromYUVData(BandBuffer* buffer, const void* y_data,
                                    const void* u_data, const void* v_data,
                                    size_t width, size_t height,
                                    size_t row_stride_y, size_t row_stride_uv,
                                    size_t pixel_stride_uv,
                                    BandBufferFormat buffer_format) {
  if (!buffer || !y_data || !u_data || !v_data) {
    BAND_LOG(LogSeverity::kError,
             "BandBuffer (%d) or y_data (%d) or u_data (%d) or v_data (%d) is "
             "null",
             buffer, y_data, u_data, v_data);
    return BandStatus::kBandErr;
  }

  buffer->impl =
      std::shared_ptr<band::Buffer>(band::Buffer::CreateFromYUVPlanes(
          static_cast<const unsigned char*>(y_data),
          static_cast<const unsigned char*>(u_data),
          static_cast<const unsigned char*>(v_data), width, height,
          row_stride_y, row_stride_uv, pixel_stride_uv,
          BufferFormat(buffer_format)));
  return buffer->impl ? BandStatus::kBandOk : BandStatus::kBandErr;
}

BandImageProcessorBuilder* BandImageProcessorBuilderCreate() {
  return new BandImageProcessorBuilder();
}

void BandImageProcessorBuilderDelete(BandImageProcessorBuilder* builder) {
  if (!builder) {
    BAND_LOG(LogSeverity::kError, "BandImageProcessorBuilder is null");
    return;
  }

  delete builder;
}

BandImageProcessor* BandImageProcessorBuilderBuild(
    BandImageProcessorBuilder* builder) {
  if (!builder) {
    BAND_LOG(LogSeverity::kError, "BandImageProcessorBuilder is null");
    return nullptr;
  }

  absl::StatusOr<std::unique_ptr<BufferProcessor>> status =
      builder->impl->Build();
  if (!status.ok()) {
    return nullptr;
  } else {
    return new BandImageProcessor(std::move(status.value()));
  }
}

BAND_CAPI_EXPORT BandStatus
BandAddOperator(BandImageProcessorBuilder* builder,
                BandImageProcessorBuilderField field, int count, ...) {
  if (!builder) {
    BAND_LOG(LogSeverity::kError, "BandImageProcessorBuilder is null");
    return BandStatus::kBandErr;
  }

  va_list vl;
  va_start(vl, count);
  switch (field) {
    case BandImageProcessorBuilderField::BAND_CROP: {
      if (count != 4) {
        return BandStatus::kBandErr;
      }

      int x0 = va_arg(vl, int);
      int y0 = va_arg(vl, int);
      int x1 = va_arg(vl, int);
      int y1 = va_arg(vl, int);
      builder->impl->AddOperation(
          std::make_unique<buffer::Crop>(x0, y0, x1, y1));
      break;
    }
    case BandImageProcessorBuilderField::BAND_RESIZE: {
      if (count != 2) {
        return BandStatus::kBandErr;
      }

      int width = va_arg(vl, int);
      int height = va_arg(vl, int);
      builder->impl->AddOperation(
          std::make_unique<buffer::Resize>(width, height));
      break;
    }
    case BandImageProcessorBuilderField::BAND_ROTATE: {
      if (count != 1) {
        return BandStatus::kBandErr;
      }

      int angle = va_arg(vl, int);
      builder->impl->AddOperation(std::make_unique<buffer::Rotate>(angle));
      break;
    }
    case BandImageProcessorBuilderField::BAND_FLIP: {
      if (count != 2) {
        return BandStatus::kBandErr;
      }

      bool horizontal = va_arg(vl, int);
      bool vertical = va_arg(vl, int);
      builder->impl->AddOperation(
          std::make_unique<buffer::Flip>(horizontal, vertical));
      break;
    }
    case BandImageProcessorBuilderField::BAND_COLOR_SPACE_CONVERT: {
      if (count != 1) {
        return BandStatus::kBandErr;
      }

      int format = va_arg(vl, int);
      builder->impl->AddOperation(
          std::make_unique<buffer::ColorSpaceConvert>(BufferFormat(format)));
      break;
    }
    case BandImageProcessorBuilderField::BAND_NORMALIZE: {
      if (count != 2) {
        return BandStatus::kBandErr;
      }

      float mean = va_arg(vl, double);
      float std = va_arg(vl, double);
      builder->impl->AddOperation(
          std::make_unique<buffer::Normalize>(mean, std, false));
      break;
    }
    case BandImageProcessorBuilderField::BAND_DATA_TYPE_CONVERT: {
      if (count != 0) {
        return BandStatus::kBandErr;
      }

      builder->impl->AddOperation(std::make_unique<buffer::DataTypeConvert>());
      break;
    }
  }
  va_end(vl);

  return BandStatus::kBandOk;
}

BandStatus BandImageProcessorProcess(BandImageProcessor* image_processor,
                                     BandBuffer* buffer,
                                     BandTensor* target_tensor) {
  if (!image_processor || !buffer || !target_tensor) {
    BAND_LOG(LogSeverity::kError,
             "BandImageProcessor (%d) or BandBuffer (%d) or BandTensor (%d) is "
             "null",
             image_processor, buffer, target_tensor);
    return BandStatus::kBandErr;
  }

  std::shared_ptr<Buffer> tensor_buffer(
      Buffer::CreateFromTensor(target_tensor->impl.get()));
  absl::Status status =
      image_processor->impl->Process(*buffer->impl.get(), *tensor_buffer.get());
  return status.ok() ? BandStatus::kBandOk : BandStatus::kBandErr;
}

void BandImageProcessorDelete(BandImageProcessor* processor) {
  delete processor;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus