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
    case DataType::NoType:
      return 0;
    case DataType::Float32:
      return sizeof(float);
    case DataType::Int32:
      return sizeof(int32_t);
    case DataType::UInt8:
      return sizeof(uint8_t);
    case DataType::Int8:
      return sizeof(int8_t);
    case DataType::Int16:
      return sizeof(int16_t);
    case DataType::Int64:
      return sizeof(int64_t);
    case DataType::String:
      return sizeof(char);
    case DataType::Bool:
      return sizeof(bool);
    case DataType::Complex64:
      return sizeof(double);
    case DataType::Float16:
      return sizeof(float) / 2;
    case DataType::Float64:
      return sizeof(double);
  }

  BAND_LOG_PROD(BAND_LOG_ERROR, "Unsupported data type : %s",
                band::GetName(GetType()).c_str());
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