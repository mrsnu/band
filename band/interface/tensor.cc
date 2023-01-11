#include "band/interface/tensor.h"

#include <cstring>

#include "band/logger.h"
#include "tensor.h"

namespace Band {
namespace Interface {

std::vector<int> ITensor::GetDimsVector() const {
  return std::vector<int>(GetDims(), GetDims() + GetNumDims());
}

bool ITensor::operator==(const ITensor& rhs) const {
  if ((GetType() == rhs.GetType()) && (GetDimsVector() == rhs.GetDimsVector()))
    return true;
  else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "%s %s != %s %s", GetName(),
                  BandTypeGetName(GetType()), rhs.GetName(),
                  BandTypeGetName(rhs.GetType()));
    return false;
  }
}

BandStatus ITensor::CopyDataFrom(const ITensor& rhs) {
  if (*this == rhs) {
    memcpy(GetData(), rhs.GetData(), GetBytes());
    return kBandOk;
  } else {
    return kBandError;
  }
}

BandStatus ITensor::CopyDataFrom(const ITensor* rhs) {
  if (!rhs) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Tried to copy null tensor");
    return kBandError;
  }

  return CopyDataFrom(*rhs);
}
}  // namespace Interface
}  // namespace Band