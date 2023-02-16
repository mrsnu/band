
#ifndef BAND_BACKEND_TFL_UTIL_H_
#define BAND_BACKEND_TFL_UTIL_H_

#include "band/common.h"
#include "band/interface/backend.h"

namespace Band {
namespace TfLite {
absl::Status Getabsl::Status(TfLiteStatus status);
DataType GetBandType(TfLiteType type);

class TfLiteUtil : public Interface::IBackendUtil {
 public:
  std::set<DeviceFlags> GetAvailableDevices() const override;
};

}  // namespace TfLite
}  // namespace Band

#endif