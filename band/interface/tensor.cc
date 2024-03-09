#include "band/interface/tensor.h"

#include <cstring>

#include "band/logger.h"
#include "tensor.h"

namespace band {
namespace interface {

bool ITensor::operator==(const ITensor& rhs) const {
  if (GetType() != rhs.GetType()) {
    return false;
  }

  if (GetDimsVector() != rhs.GetDimsVector()) {
    return false;
  }

  return true;
}

bool ITensor::operator!=(const ITensor& rhs) const { return !(*this == rhs); }

size_t ITensor::GetBytes() const {
  return GetDataTypeBytes(GetType()) * GetNumElements();
}

size_t ITensor::GetNumElements() const {
  size_t num_elements = 1;
  for (auto dim : GetDimsVector()) {
    num_elements *= dim;
  }
  return num_elements;
}

std::vector<int> ITensor::GetDimsVector() const {
  return std::vector<int>(GetDims(), GetDims() + GetNumDims());
}

absl::Status ITensor::CopyDataFrom(const ITensor& rhs) {
  if (*this != rhs) {
    return absl::InternalError("");
  }
  memcpy(GetData(), rhs.GetData(), GetBytes());
  return absl::OkStatus();
}

absl::Status ITensor::CopyDataFrom(const ITensor* rhs) {
  if (!rhs) {
    return absl::InternalError("Tried to copy null tensor");
  }

  return CopyDataFrom(*rhs);
}
}  // namespace interface
}  // namespace band