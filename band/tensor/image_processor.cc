
#include "band/tensor/image_processor.h"

#include "band/tensor/buffer.h"
#include "band/tensor/image_operation.h"

namespace band {
namespace tensor {

absl::StatusOr<std::unique_ptr<ImageProcessor>> ImageProcessorBuilder::Build(
    const Buffer* input, Buffer* output) {
  bool requires_validation = input != nullptr && output != nullptr;

  std::unique_ptr<ImageProcessor> processor(new ImageProcessor());

  for (auto& operation : operations_) {
    processor->operations_.push_back(operation->Clone());
  }

  // special case: resize input to output if no operations are specified
  if (operations_.empty() && requires_validation) {
    processor->operations_.push_back(
        new ResizeOperation(output->GetDimension()));
  }

  if (requires_validation) {
    absl::Status status = processor->Process(*input);
    if (!status.ok()) {
      return status;
    }
  }

  return std::move(processor);
}

}  // namespace tensor
}  // namespace band