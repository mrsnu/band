#ifndef BAND_BUFFER_IMAGE_PROCESSOR_H
#define BAND_BUFFER_IMAGE_PROCESSOR_H

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/buffer/processor.h"
#include "band/common.h"

namespace band {

class IOperation;
class Processor;

class ImageProcessorBuilder : public IProcessorBuilder {
 public:
  virtual absl::StatusOr<std::unique_ptr<Processor>> Build(
      const Buffer* input = nullptr, Buffer* output = nullptr) override;

  // TODO(dostos): type check for image operations
  using IProcessorBuilder::AddOperation;
};
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_PROCESSOR_H