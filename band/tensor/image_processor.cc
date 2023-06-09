
#include "band/tensor/image_processor.h"

#include "band/tensor/buffer.h"
#include "band/tensor/image_operation.h"

namespace band {
namespace tensor {

absl::StatusOr<std::unique_ptr<IProcessor>> ImageProcessorBuilder::Build(
    const Buffer* input, Buffer* output) {
  bool requires_validation = input != nullptr && output != nullptr;

  std::vector<IOperation*> operations;
  for (auto& operation : operations_) {
    operations.push_back(operation->Clone());
  }
  // special case: resize input to output if no operations are specified
  if (operations_.empty() && requires_validation) {
    operations.push_back(new ResizeOperation(output->GetDimension()));
  }
  std::unique_ptr<IProcessor> processor = CreateProcessor(operations);
  if (requires_validation) {
    absl::Status status = processor->Process(*input, *output);
    if (!status.ok()) {
      return status;
    }
  }
  return std::move(processor);
}

}  // namespace tensor
}  // namespace band