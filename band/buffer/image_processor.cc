
#include "band/buffer/image_processor.h"

#include "band/buffer/buffer.h"
#include "band/buffer/image_operation.h"

namespace band {

absl::StatusOr<std::unique_ptr<Processor>> ImageProcessorBuilder::Build(
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
  std::unique_ptr<Processor> processor = CreateProcessor(operations);
  if (requires_validation) {
    absl::Status status = processor->Process(*input, *output);
    if (!status.ok()) {
      return status;
    }
  }
  return std::move(processor);
}

}  // namespace band