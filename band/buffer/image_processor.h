#ifndef BAND_BUFFER_IMAGE_PROCESSOR_H_
#define BAND_BUFFER_IMAGE_PROCESSOR_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/buffer/buffer_processor.h"
#include "band/common.h"

namespace band {

class IBufferOperator;
class BufferProcessor;

class ImageProcessorBuilder : public IBufferProcessorBuilder {
 public:
  virtual absl::StatusOr<std::unique_ptr<BufferProcessor>> Build() override;
};

class ImageProcessor : public BufferProcessor {
 public:
  virtual ~ImageProcessor();
  virtual absl::Status Process(const Buffer& input, Buffer& output);

 private:
  std::unique_ptr<AutoConvert> auto_convert_cache_;
};
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_PROCESSOR_H