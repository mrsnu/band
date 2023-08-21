#ifndef BAND_DEVICE_UTIL_H_
#define BAND_DEVICE_UTIL_H_

#include <fstream>
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

template <typename T>
absl::StatusOr<T> TryRead(std::vector<std::string> paths,
                          std::vector<float> multipliers = {});

absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths,
                                    std::vector<float> multipliers = {1.f});
absl::StatusOr<double> TryReadDouble(std::vector<std::string> paths,
                                     std::vector<float> multipliers = {1.f});

absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths, std::vector<float> multipliers = {1.f});

std::vector<std::string> ListFilesInPath(const char* path);
std::vector<std::string> ListFilesInPathPrefix(const char* path,
                                               const char* prefix);
std::vector<std::string> ListFilesInPathSuffix(const char* path,
                                               const char* suffix);
std::vector<std::string> ListDirectoriesInPath(const char* path);
std::vector<std::string> ListDirectoriesInPathPrefix(const char* path,
                                                     const char* prefix);
std::vector<std::string> ListDirectoriesInPathSuffix(const char* path,
                                                     const char* suffix);

bool IsFileAvailable(std::string path);
void Root();
std::string RunCommand(const std::string& command);
absl::StatusOr<std::string> GetDeviceProperty(const std::string& property);

}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
