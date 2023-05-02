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
  } else if (status == kTfLiteDelegateError) {
    return absl::UnavailableError("TfLite Delegate Error");
  }else {
    return absl::InternalError("TfLite Error");
  }
}

DataType GetBandType(TfLiteType type) { return DataType(type); }
std::set<DeviceFlags> TfLiteUtil::GetAvailableDevices() const {
  static std::set<DeviceFlags> valid_devices = {};
  static std::once_flag once_flag;

  std::call_once(once_flag, [&]() {
    for (size_t flag = 0; flag < GetSize<DeviceFlags>(); flag++) {
      const DeviceFlags device_flag = static_cast<DeviceFlags>(flag);
      if (TfLiteModelExecutor::GetDeviceDelegate(device_flag).ok()) {
        valid_devices.insert(device_flag);
      }
    }
  });

  return valid_devices;
}
}  // namespace tfl
}  // namespace band
