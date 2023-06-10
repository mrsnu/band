#ifndef BAND_BUFFER_IMAGE_PROCESSOR_H
#define BAND_BUFFER_IMAGE_PROCESSOR_H

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/buffer/buffer_processor.h"
#include "band/common.h"

namespace band {

class IOperation;
class BufferProcessor;

class ImageProcessorBuilder : public IBufferProcessorBuilder {
 public:
  virtual absl::StatusOr<std::unique_ptr<BufferProcessor>> Build(
      const Buffer* input = nullptr, Buffer* output = nullptr) override;

  // TODO(dostos): type check for image operations
  using IBufferProcessorBuilder::AddOperation;
};
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_PROCESSOR_H