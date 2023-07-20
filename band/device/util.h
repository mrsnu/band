#ifndef BAND_DEVICE_UTIL_H_
#define BAND_DEVICE_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"

#if defined(__ANDROID__) || defined(__IOS__)
#define BAND_IS_MOBILE 1
#else
#define BAND_IS_MOBILE 0
#endif

namespace band {
namespace device {
bool IsRooted();
std::string RunCommand(const std::string& command);
absl::StatusOr<std::string> GetDeviceProperty(const std::string& property);
}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
