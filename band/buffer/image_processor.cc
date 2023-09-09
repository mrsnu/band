#include "band/buffer/image_processor.h"

#include "absl/strings/str_format.h"
#include "band/buffer/buffer.h"
#include "band/buffer/common_operator.h"
#include "band/buffer/image_operator.h"
#include "image_processor.h"
namespace band {
using namespace buffer;

absl::StatusOr<std::unique_ptr<BufferProcessor>>
ImageProcessorBuilder::Build() {
  std::vector<IBufferOperator*> operations;
  // TODO(dostos): reorder operations to optimize performance
  // e.g., crop before color conversion

  for (size_t i = 0; i < operations_.size(); ++i) {
    for (size_t j = 1; j < operations_.size(); ++j) {
      if (i == j) {
        continue;
      }
      IBufferOperator& lhs = *operations_[i];
      IBufferOperator& rhs = *operations_[j];
      if (typeid(lhs) == typeid(rhs)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("operation %s is duplicated.", typeid(lhs).name()));
      }
    }
  }

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

  // by default, automatically convert the color space and resize the entire
  // image.
  if (operations.empty()) {
    operations.push_back(new AutoConvert());
  }

  return std::move(CreateProcessor(operations));
}
}  // namespace band