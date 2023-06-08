#include "band/tensor/operation.h"

#include "operation.h"

namespace band {
namespace tensor {

IOperation::IOperation(const IOperation& rhs)
    : output_(std::make_unique<Buffer>(rhs.output_.get())) {}

void IOperation::SetOutput(Buffer* output) {
  output_ = std::make_unique<Buffer>(output, [](Buffer* buffer) {});
}

}  // namespace tensor
}  // namespace band
