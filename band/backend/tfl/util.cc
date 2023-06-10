#include "band/backend/tfl/util.h"

#include <mutex>
#include <set>

#include "band/backend/tfl/model_executor.h"
#include "band/logger.h"

namespace band {
namespace tfl {

absl::Status GetBandStatus(TfLiteStatus status) { 
  if (status == kTfLiteOk) {
    return absl::OkStatus();
  } else {
    return absl::InternalError("TfLite Error");
  }
}

DataType GetBandDataType(TfLiteType type) { return DataType(type); }
std::set<DeviceFlag> TfLiteUtil::GetAvailableDevices() const {
  static std::set<DeviceFlag> valid_devices = {};
  static std::once_flag once_flag;

  std::call_once(once_flag, [&]() {
    for (size_t flag = 0; flag < DeviceFlag::kBandNumDeviceFlag; flag++) {
      const DeviceFlag device_flag = static_cast<DeviceFlag>(flag);
      if (TfLiteModelExecutor::GetDeviceDelegate(device_flag).ok()) {
        valid_devices.insert(device_flag);
      }
    }
  });

  return valid_devices;
}
}  // namespace tfl
}  // namespace band
