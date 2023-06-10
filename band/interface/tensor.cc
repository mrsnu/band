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
    case DataType::kBandNoType:
      return 0;
    case DataType::kBandFloat32:
      return sizeof(float);
    case DataType::kBandInt32:
      return sizeof(int32_t);
    case DataType::kBandUInt8:
      return sizeof(uint8_t);
    case DataType::kBandInt8:
      return sizeof(int8_t);
    case DataType::kBandInt16:
      return sizeof(int16_t);
    case DataType::kBandInt64:
      return sizeof(int64_t);
    case DataType::kBandString:
      return sizeof(char);
    case DataType::kBandBool:
      return sizeof(bool);
    case DataType::kBandComplex64:
      return sizeof(double);
    case DataType::kBandFloat16:
      return sizeof(float) / 2;
    case DataType::kBandFloat64:
      return sizeof(double);
    default:
      break;
  }

  BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported data type : %s",
                band::ToString(GetType()).c_str());
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