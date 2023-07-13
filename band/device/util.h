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
// Read from the first available path, return absl::NotFoundError if none of the
// paths exist.
absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths);
absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths);
absl::StatusOr<std::string> TryReadString(std::vector<std::string> paths);

bool IsRooted();
std::string RunCommand(const std::string& command);
absl::StatusOr<std::string> GetDeviceProperty(const std::string& property);
// Get preset devfreq paths for known devices.
std::map<DeviceFlag, std::string> GetDevfreqPaths();

}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
