#include "band/backend/tfl/util.h"

#include <mutex>
#include <set>

#include "band/backend/tfl/interpreter.h"
#include "band/logger.h"

namespace Band {
namespace TfLite {

absl::Status GetBandStatus(TfLiteStatus status) { 
  switch (status) {
    case kTfLiteOk: {
      return absl::OkStatus();
    } break;
    case kTfLiteError: {
      return absl::UnknownError("TfLiteError occurs.");
    } break;
    case kTfLiteDelegateError: {
      return absl::ResourceExhaustedError("TfLite Delegate failed.");
    } break;
  }
}
BandType GetBandType(TfLiteType type) { return BandType(type); }
std::set<BandDeviceFlags> TfLiteUtil::GetAvailableDevices() const {
  static std::set<BandDeviceFlags> valid_devices = {};
  static std::once_flag once_flag;

  std::call_once(once_flag, [&]() {
    TfLiteInterpreter interpreter;
    for (int flag = 0; flag < kBandNumDevices; flag++) {
      const BandDeviceFlags device_flag = static_cast<BandDeviceFlags>(flag);
      if (interpreter.GetDeviceDelegate(device_flag).ok()) {
        valid_devices.insert(device_flag);
      }
    }
  });

  return valid_devices;
}
}  // namespace TfLite
}  // namespace Band
