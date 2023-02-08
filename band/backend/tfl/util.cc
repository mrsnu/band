#include "band/backend/tfl/util.h"

#include <mutex>
#include <set>

#include "band/backend/tfl/model_executor.h"
#include "band/logger.h"

namespace Band {
namespace TfLite {

BandStatus GetBandStatus(TfLiteStatus status) { return BandStatus(status); }
DataType GetBandType(TfLiteType type) { return DataType(type); }
std::set<DeviceFlags> TfLiteUtil::GetAvailableDevices() const {
  static std::set<DeviceFlags> valid_devices = {};
  static std::once_flag once_flag;

  std::call_once(once_flag, [&]() {
    for (int flag = 0; flag < kBandNumDevices; flag++) {
      const DeviceFlags device_flag = static_cast<DeviceFlags>(flag);
      if (TfLiteModelExecutor::GetDeviceDelegate(device_flag).first ==
          kBandOk) {
        valid_devices.insert(device_flag);
      }
    }
  });

  return valid_devices;
}
}  // namespace TfLite
}  // namespace Band
