#include "band/interface/tensor.h"

#include <cstring>

#include "band/logger.h"

namespace Band {
namespace Interface {

BandStatus ITensor::CopyDataFrom(const ITensor& rhs) {
  if (GetType() != rhs.GetType() && GetDims() != rhs.GetDims()) {
    return kBandError;
  }

  memcpy(GetData(), rhs.GetData(), GetBytes());
  return kBandOk;
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