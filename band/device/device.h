#ifndef BAND_DEVICE_DEVICE_H_
#define BAND_DEVICE_DEVICE_H_

#include <mutex>

#include "band/device/util.h"

namespace band {
namespace device {

void LoadDeviceInfo() {
  static std::once_flag device_flag;
  std::call_once(device_flag, [] {
#if BAND_IS_MOBILE
    // Get thermal_zone info
    // Get cpufreq, devfreq info
    if (IsRooted()) {
      BAND_LOG_INTERNAL(BAND_LOG_INFO, "Device is rooted");
      
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_INFO,
                        "Device is not rooted. Thermal and frequency "
                        "information cannot be obtained.");
    }
#endif  // BAND_IS_MOBILE
  });
}

}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_DEVICE_H_