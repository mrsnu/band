#include "band/tensor/operation.h"

#include "operation.h"

namespace band {
namespace tensor {

absl::Status tensor::CropOperation::Process(const Buffer* input) {
  return absl::Status();
}

bool CropOperation::IsCompatible(const Buffer* input) const { return false; }

absl::Status ResizeOperation::Process(const Buffer* input) {
  return absl::Status();
}

bool ResizeOperation::IsCompatible(const Buffer* input) const { return false; }

absl::Status ConvertOperation::Process(const Buffer* input) {
  return absl::Status();
}

bool ConvertOperation::IsCompatible(const Buffer* input) const { return false; }

}  // namespace tensor
}  // namespace band