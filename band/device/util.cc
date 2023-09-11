// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/device/util.h"

#if defined(_POSIX_VERSION)
#include <dirent.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#error "Unsupported platform"
#endif

#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#include "absl/strings/str_format.h"
#include "band/logger.h"

namespace band {
namespace device {

template <typename T>
absl::StatusOr<T> TryRead(std::vector<std::string> paths,
                          std::vector<float> multipliers) {
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
    std::ifstream fs(path, std::ifstream::binary);
    if (fs.is_open()) {
      T output;
      fs >> output;
      return output * multipliers[i];
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No available path: %s, %s", paths[0].c_str(), strerror(errno)));
}

absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths,
                                    std::vector<float> multipliers) {
  return TryRead<size_t>(paths, multipliers);
}

absl::StatusOr<double> TryReadDouble(std::vector<std::string> paths,
                                     std::vector<float> multipliers) {
  return TryRead<double>(paths, multipliers);
}

absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths, std::vector<float> multipliers) {
  if (paths.size() != multipliers.size()) {
    return absl::InternalError(
        "Number of paths and multipliers must be the same.");
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto path = paths[i];
    std::ifstream fs(path, std::ifstream::binary);
    if (fs.is_open()) {
      std::vector<size_t> outputs;
      size_t output;
      while (fs >> output) {
        outputs.push_back(output * multipliers[i]);
      }
      return outputs;
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No available path: %s, %s", paths[0].c_str(), strerror(errno)));
}

absl::StatusOr<std::vector<double>> TryReadDoubles(
    std::vector<std::string> paths, std::vector<float> multipliers) {
  if (paths.size() != multipliers.size()) {
    return absl::InternalError(
        "Number of paths and multipliers must be the same.");
  }

  for (size_t i = 0; i < paths.size(); i++) {
    auto path = paths[i];
    std::ifstream fs(path, std::ifstream::binary);
    if (fs.is_open()) {
      std::vector<double> outputs;
      double output;
      while (fs >> output) {
        outputs.push_back(output * multipliers[i]);
      }
      return outputs;
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No available path: %s, %s", paths[0].c_str(), strerror(errno)));
}

std::vector<std::string> ListFilesInPath(const char* path) {
  std::vector<std::string> ret;

#if defined(_POSIX_VERSION)
  DIR* dir = opendir(path);
  if (dir == nullptr) {
    return {};
  }
  struct dirent* entry = readdir(dir);

  while (entry != nullptr) {
    if (entry->d_type == DT_REG) {
      ret.push_back(entry->d_name);
    }
    entry = readdir(dir);
  }
  closedir(dir);
#elif defined(_WIN32)
  WIN32_FIND_DATAA find_data;
  HANDLE find_handle =
      FindFirstFileA((std::string(path) + "/*").c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    return {};
  }
  do {
    if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      continue;
    }
    ret.push_back(find_data.cFileName);
  } while (FindNextFileA(find_handle, &find_data));
  FindClose(find_handle);
#endif

  return ret;
}

std::vector<std::string> ListFilesInPathPrefix(const char* path,
                                               const char* prefix) {
  std::vector<std::string> ret;
  for (const auto& file : ListFilesInPath(path)) {
    if (file.find(prefix) == 0) {
      ret.push_back(file);
    }
  }
  return ret;
}

std::vector<std::string> ListFilesInPathSuffix(const char* path,
                                               const char* suffix) {
  std::vector<std::string> ret;
  for (const auto& file : ListFilesInPath(path)) {
    if (file.find(suffix) == file.size() - strlen(suffix)) {
      ret.push_back(file);
    }
  }
  return ret;
}

std::vector<std::string> ListDirectoriesInPath(const char* path) {
  std::vector<std::string> ret;
#if defined(_POSIX_VERSION)
  DIR* dir = opendir(path);
  if (dir == nullptr) {
    return {};
  }
  struct dirent* entry = readdir(dir);

  while (entry != nullptr) {
    if (entry->d_type == DT_DIR || entry->d_type == DT_LNK) {
      ret.push_back(entry->d_name);
    }
    entry = readdir(dir);
  }
  closedir(dir);
#elif defined(_WIN32)
  // list all directories in path
  WIN32_FIND_DATAA find_data;
  HANDLE find_handle =
      FindFirstFileA((std::string(path) + "/*").c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    return {};
  }
  do {
    if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
      ret.push_back(find_data.cFileName);
    }
  } while (FindNextFileA(find_handle, &find_data));
  FindClose(find_handle);
#endif
  return ret;
}

std::vector<std::string> ListDirectoriesInPathPrefix(const char* path,
                                                     const char* prefix) {
  std::vector<std::string> ret;
  for (const auto& file : ListDirectoriesInPath(path)) {
    if (file.find(prefix) == 0) {
      ret.push_back(file);
    }
  }
  return ret;
}

std::vector<std::string> ListDirectoriesInPathSuffix(const char* path,
                                                     const char* suffix) {
  std::vector<std::string> ret;
  for (const auto& file : ListDirectoriesInPath(path)) {
    if (file.find(suffix) == file.size() - strlen(suffix)) {
      ret.push_back(file);
    }
  }
  return ret;
}

bool IsFileAvailable(std::string path) {
#if defined(_POSIX_VERSION)
  return access(path.c_str(), F_OK) != -1;
#elif defined(_WIN32)
  return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
#endif
}

std::string RunCommand(const std::string& command) {
  std::string result = "";
  // suppress stderr
  FILE* pipe = popen((command + "2>&1").c_str(), "r");
  if (pipe != nullptr) {
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
    pclose(pipe);
  }
  return result;
}

void Root() {
  bool is_rooted = true;
  is_rooted =
      RunCommand("su -c 'echo rooted'").find("rooted") != std::string::npos;
  BAND_LOG_INTERNAL(BAND_LOG_INFO, "Is rooted: %d", is_rooted);
}

absl::StatusOr<std::string> GetDeviceProperty(const std::string& property) {
  static std::once_flag flag;
  static std::map<std::string, std::string> properties;

  std::call_once(flag, [&]() {
    std::string output = RunCommand("getprop");
    std::stringstream ss(output);
    std::string line;
    while (std::getline(ss, line, '\n')) {
      size_t pos = line.find('[');
      if (pos == std::string::npos) {
        continue;
      }
      std::string key = line.substr(0, pos);
      std::string value = line.substr(pos + 1, line.size() - pos - 2);
      properties[key] = value;
    }
  });

  if (properties.find(property) != properties.end()) {
    return properties[property];
  } else {
    return absl::NotFoundError("Property not found");
  }
}

}  // namespace device
}  // namespace band