#include "band/interface/tensor.h"

#include <cstring>

#include "band/logger.h"

namespace Band {
namespace Interface {

absl::Status ITensor::CopyDataFrom(const ITensor& rhs) {
  if (GetType() != rhs.GetType() && GetDims() != rhs.GetDims()) {
    return absl::InvalidArgumentError(
        "lhs and rhs must have the same dimension and type.");
  }

  memcpy(GetData(), rhs.GetData(), GetBytes());
  return absl::OkStatus();
}

absl::Status ITensor::CopyDataFrom(const ITensor* rhs) {
  if (!rhs) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Tried to copy null tensor");
    return absl::InvalidArgumentError("Tried to copy null tensor.");
  }

  return CopyDataFrom(*rhs);
}
}  // namespace Interface
}  // namespace Band