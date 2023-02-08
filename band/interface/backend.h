#ifndef BAND_INTERFACE_BACKEND_H_
#define BAND_INTERFACE_BACKEND_H_

#include "band/c/common.h"

namespace Band {
namespace Interface {
class IBackendSpecific {
 public:
  virtual BackendType GetBackendType() const = 0;
  bool IsCompatible(const IBackendSpecific& rhs) const {
    return IsCompatible(&rhs);
  }
  bool IsCompatible(const IBackendSpecific* rhs) const {
    return GetBackendType() == rhs->GetBackendType();
  }
};

class IBackendUtil {
 public:
  virtual std::set<DeviceFlags> GetAvailableDevices() const = 0;
};

}  // namespace Interface
}  // namespace Band

#endif  // BAND_INTERFACE_BACKEND_H_