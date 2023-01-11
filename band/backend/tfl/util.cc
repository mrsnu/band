#include "band/backend/tfl/util.h"

#include <mutex>
#include <set>

#include "band/backend/tfl/interpreter.h"
#include "band/logger.h"

namespace Band {
namespace TfLite {

BandStatus GetBandStatus(TfLiteStatus status) { return BandStatus(status); }
BandType GetBandType(TfLiteType type) { return BandType(type); }
std::set<BandDeviceFlags> TfLiteUtil::GetAvailableDevices() const {
  static std::set<BandDeviceFlags> valid_devices = {};
  static std::once_flag once_flag;

  std::call_once(once_flag, [&]() {
    for (int flag = 0; flag < kBandNumDevices; flag++) {
      const BandDeviceFlags device_flag = static_cast<BandDeviceFlags>(flag);
      if (TfLiteInterpreter::GetDeviceDelegate(device_flag).first == kBandOk) {
        valid_devices.insert(device_flag);
      }
    }
  });

  return valid_devices;
}
}  // namespace TfLite
}  // namespace Band
