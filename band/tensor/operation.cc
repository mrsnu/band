#include "band/tensor/operation.h"

#include "operation.h"

namespace band {
namespace tensor {

void IOperation::SetOutput(Buffer* output) {
  output_ = std::shared_ptr<Buffer>(output, [](Buffer* buffer) {});
}

}  // namespace tensor
}  // namespace band
