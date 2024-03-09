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
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_PROCESSOR_H