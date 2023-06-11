
#include "band/buffer/image_processor.h"

#include "absl/strings/str_format.h"
#include "band/buffer/buffer.h"
#include "band/buffer/image_operator.h"
#include "image_processor.h"
namespace band {
using namespace buffer;

absl::StatusOr<std::unique_ptr<BufferProcessor>>
ImageProcessorBuilder::Build() {
  std::vector<IBufferOperator*> operations;
  // TODO(dostos): reorder operations to optimize performance
  // e.g., crop before color conversion
  for (auto& operation : operations_) {
    if (operation == nullptr) {
      return absl::InvalidArgumentError("operation is nullptr.");
    }

    if (operation->GetOpType() != IBufferOperator::Type::kImage &&
        operation->GetOpType() != IBufferOperator::Type::kCommon) {
      return absl::InvalidArgumentError(
          absl::StrFormat("operation type %d is not supported.",
                          static_cast<int>(operation->GetOpType())));
    }

    operations.push_back(operation->Clone());
  }
  // special case: resize input to output if no operations are specified
  if (operations_.empty()) {
    operations.push_back(new Resize());
  }
  return std::move(CreateProcessor(operations));
}
}  // namespace band