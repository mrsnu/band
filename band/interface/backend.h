#ifndef BAND_INTERFACE_BACKEND_H_
#define BAND_INTERFACE_BACKEND_H_

#include "band/c/common.h"

namespace band {
namespace interface {
class IBackendSpecific {
 public:
  virtual BandBackendType GetBackendType() const = 0;
  bool IsCompatible(const IBackendSpecific& rhs) const {
    return IsCompatible(&rhs);
  }
  bool IsCompatible(const IBackendSpecific* rhs) const {
    return GetBackendType() == rhs->GetBackendType();
  }
};

class IBackendUtil {
 public:
  virtual std::set<BandDeviceFlags> GetAvailableDevices() const = 0;
};

}  // namespace interface
}  // namespace band

#endif  // BAND_INTERFACE_BACKEND_H_