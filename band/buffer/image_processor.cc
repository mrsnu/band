
#include "band/buffer/image_processor.h"

#include "band/buffer/buffer.h"
#include "band/buffer/image_operation.h"
#include "image_processor.h"

namespace band {

absl::StatusOr<std::unique_ptr<BufferProcessor>>
ImageProcessorBuilder::Build() {
  std::vector<IOperation*> operations;
  // TODO(dostos): reorder operations to optimize performance
  // e.g., crop before color conversion
  for (auto& operation : operations_) {
    operations.push_back(operation->Clone());
  }
  // special case: resize input to output if no operations are specified
  if (operations_.empty()) {
    operations.push_back(new ResizeOperation());
  }
  return std::move(CreateProcessor(operations));
}

absl::Status ImageProcessorBuilder::AddOperation(
    std::unique_ptr<IOperation> operation) {
  if (operation == nullptr) {
    return absl::InvalidArgumentError("operation is nullptr.");
  }

  if (operation->GetOperationType() != IOperation::OperationType::kImage) {
    return absl::InvalidArgumentError("operation is not an image operation.");
  }

  return absl::OkStatus();
}  // namespace band
}  // namespace band