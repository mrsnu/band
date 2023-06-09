#include "band/buffer/operation.h"

#include "operation.h"

namespace band {
void IOperation::SetOutput(Buffer* output) {
  output_ = std::shared_ptr<Buffer>(output, [](Buffer* buffer) {});
}
}  // namespace band
