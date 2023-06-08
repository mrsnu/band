#ifndef BAND_TENSOR_IMAGE_PROCESSOR_H
#define BAND_TENSOR_IMAGE_PROCESSOR_H

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/tensor/processor.h"

namespace band {
namespace tensor {

class IOperation;
class ImageProcessor;

class ImageProcessorBuilder : public IProcessorBuilder<ImageProcessor> {
 public:
  virtual absl::StatusOr<std::unique_ptr<ImageProcessor>> Build(
      const Buffer* input = nullptr, Buffer* output = nullptr) override;

  // TODO(dostos): type check for image operations
  using IProcessorBuilder<ImageProcessor>::AddOperation;
};

class ImageProcessor : public IProcessor {
 public:
  absl::Status Process(const Buffer& input);
  absl::Status IsValid(const Buffer& input) const;

 protected:
  friend class ImageProcessorBuilder;
};

}  // namespace tensor
}  // namespace band

#endif  // BAND_TENSOR_IMAGE_PROCESSOR_H