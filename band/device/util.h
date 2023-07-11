#ifndef BAND_DEVICE_UTIL_H_
#define BAND_DEVICE_UTIL_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#ifdef __ANDROID__
#define BAND_SUPPORT_DEVICE 1
#else
#define BAND_SUPPORT_DEVICE 0
#endif

namespace band {
namespace device {
// Read from the first available path, return absl::NotFoundError if none of the
// paths exist.
absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths);
absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths);
absl::StatusOr<std::string> TryReadString(std::vector<std::string> paths);

bool SupportsDevice();
bool IsRooted();

}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
