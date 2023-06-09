#ifndef BAND_TENSOR_IMAGE_PROCESSOR_H
#define BAND_TENSOR_IMAGE_PROCESSOR_H

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/tensor/processor.h"

namespace band {
namespace tensor {

class IOperation;
class IProcessor;

class ImageProcessorBuilder : public IProcessorBuilder {
 public:
  virtual absl::StatusOr<std::unique_ptr<IProcessor>> Build(
      const Buffer* input = nullptr, Buffer* output = nullptr) override;

  // TODO(dostos): type check for image operations
  using IProcessorBuilder::AddOperation;
};

}  // namespace tensor
}  // namespace band

#endif  // BAND_TENSOR_IMAGE_PROCESSOR_H