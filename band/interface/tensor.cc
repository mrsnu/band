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

size_t ITensor::GetBytes() const { return GetPixelBytes() * GetNumElements(); }

size_t ITensor::GetPixelBytes() const {
  switch (GetType()) {
    case DataType::kNoType:
      return 0;
    case DataType::kFloat32:
      return sizeof(float);
    case DataType::kInt32:
      return sizeof(int32_t);
    case DataType::kUInt8:
      return sizeof(uint8_t);
    case DataType::kInt8:
      return sizeof(int8_t);
    case DataType::kInt16:
      return sizeof(int16_t);
    case DataType::kInt64:
      return sizeof(int64_t);
    case DataType::kString:
      return sizeof(char);
    case DataType::kBool:
      return sizeof(bool);
    case DataType::kComplex64:
      return sizeof(double);
    case DataType::kFloat16:
      return sizeof(float) / 2;
    case DataType::kFloat64:
      return sizeof(double);
    default:
      break;
  }

  BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported data type : %s",
                ToString(GetType()));
  return 0;
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