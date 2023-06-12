#include "band/c/c_api_buffer.h"

#include "band/buffer/buffer.h"
#include "band/buffer/buffer_processor.h"
#include "band/buffer/common_operator.h"
#include "band/buffer/image_operator.h"
#include "band/c/c_api_internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

using namespace band;

BandBuffer* BandBufferCreate() { return new BandBuffer(); }

void BandBufferDelete(BandBuffer* buffer) { delete buffer; }

BandStatus BandBufferSetFromRawData(BandBuffer* buffer, const void* data,
                                    size_t width, size_t height,
                                    BandBufferFormat format) {
  buffer->impl = std::shared_ptr<band::Buffer>(
      band::Buffer::CreateFromRaw(static_cast<const unsigned char*>(data),
                                  width, height, BufferFormat(format)));
  return buffer->impl ? BandStatus::kBandOk : BandStatus::kBandError;
}

BandStatus BandBufferSetFromYUVData(BandBuffer* buffer, const void* y_data,
                                    const void* u_data, const void* v_data,
                                    size_t width, size_t height,
                                    size_t row_stride_y, size_t row_stride_uv,
                                    size_t pixel_stride_uv,
                                    BandBufferFormat buffer_format) {
  buffer->impl =
      std::shared_ptr<band::Buffer>(band::Buffer::CreateFromYUVPlanes(
          static_cast<const unsigned char*>(y_data),
          static_cast<const unsigned char*>(u_data),
          static_cast<const unsigned char*>(v_data), width, height,
          row_stride_y, row_stride_uv, pixel_stride_uv,
          BufferFormat(buffer_format)));
  return buffer->impl ? BandStatus::kBandOk : BandStatus::kBandError;
}

BandImageProcessorBuilder* BandImageProcessorBuilderCreate() {
  return new BandImageProcessorBuilder();
}

void BandImageProcessorBuilderDelete(BandImageProcessorBuilder* builder) {
  delete builder;
}

void BandImageProcessorBuilderAddCrop(BandImageProcessorBuilder* builder,
                                      int x0, int y0, int x1, int y1) {
  builder->impl->AddOperation(std::make_unique<buffer::Crop>(x0, y0, x1, y1));
}

void BandImageProcessorBuilderAddResize(BandImageProcessorBuilder* builder,
                                        int width, int height) {
  builder->impl->AddOperation(std::make_unique<buffer::Resize>(width, height));
}

void BandImageProcessorBuilderAddRotate(BandImageProcessorBuilder* builder,
                                        int angle) {
  builder->impl->AddOperation(std::make_unique<buffer::Rotate>(angle));
}

void BandImageProcessorBuilderAddFlip(BandImageProcessorBuilder* builder,
                                      bool horizontal);

void BandImageProcessorBuilderAddColorSpaceConvert(
    BandImageProcessorBuilder* builder, BandBufferFormat format) {
  builder->impl->AddOperation(
      std::make_unique<buffer::ColorSpaceConvert>(BufferFormat(format)));
}

void BandImageProcessorBuilderAddNormalize(BandImageProcessorBuilder* builder,
                                           float mean, float std) {
  builder->impl->AddOperation(
      std::make_unique<buffer::Normalize>(mean, std, false));
}

void BandImageProcessorBuilderAddDataTypeConvert(
    BandImageProcessorBuilder* builder) {
  builder->impl->AddOperation(std::make_unique<buffer::DataTypeConvert>());
}

BandImageProcessor* BandImageProcessorBuilderBuild(
    BandImageProcessorBuilder* builder) {
  absl::StatusOr<std::unique_ptr<BufferProcessor>> status =
      builder->impl->Build();
  if (!status.ok()) {
    return nullptr;
  } else {
    return new BandImageProcessor(std::move(status.value()));
  }
}

BandStatus BandImageProcessorProcess(BandImageProcessor* image_processor,
                                     BandBuffer* buffer,
                                     BandTensor* target_tensor) {
  std::shared_ptr<Buffer> tensor_buffer(
      Buffer::CreateFromTensor(target_tensor->impl.get()));
  absl::Status status =
      image_processor->impl->Process(*buffer->impl.get(), *tensor_buffer.get());
  return status.ok() ? BandStatus::kBandOk : BandStatus::kBandError;
}

void BandImageProcessorDelete(BandImageProcessor* processor) {
  delete processor;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus