#ifndef BAND_DEVICE_UTIL_H_
#define BAND_DEVICE_UTIL_H_

#include <map>
#include <string>
#include <vector>
#include <fstream>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"

namespace band {
namespace device {

template <typename T>
absl::StatusOr<T> TryRead(std::vector<std::string> paths,
                          std::vector<float> multipliers = {}) {
  // get from path and multiply by multiplier
  if (multipliers.size() == 0) {
    multipliers.resize(paths.size(), 1.f);
  }

  if (paths.size() != multipliers.size()) {
    return absl::InternalError(
        "Number of paths and multipliers must be the same.");
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto path = paths[i];
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      T output;
      fs >> output;
      return output * multipliers[i];
    }
  }
  return absl::NotFoundError("No available path");
}

absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths,
                                    std::vector<float> multipliers = {}) {
  return TryRead<size_t>(paths, multipliers);
}

absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths, std::vector<float> multipliers = {}) {
  // get from path and multiply by multiplier
  if (multipliers.size() == 0) {
    multipliers.resize(paths.size(), 1.f);
  }

  if (paths.size() != multipliers.size()) {
    return absl::InternalError(
        "Number of paths and multipliers must be the same.");
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto path = paths[i];
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      std::vector<size_t> outputs;
      size_t output;
      while (fs >> output) {
        outputs.push_back(output * multipliers[i]);
      }
      return outputs;
    }
  }
  return absl::NotFoundError("No available path");
}

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
bool IsRooted();
std::string RunCommand(const std::string& command);
absl::StatusOr<std::string> GetDeviceProperty(const std::string& property);

}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
