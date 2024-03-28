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

#include "band/logger.h"
#include "util.h"

namespace band {
namespace device {

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
#ifdef _WIN32
  FILE* pipe = _popen((command + "2>&1").c_str(), "r");
#else
  FILE* pipe = popen((command + "2>&1").c_str(), "r");
#endif
  if (pipe != nullptr) {
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result += buffer;
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
  }
  return result;
}

bool IsRooted() {
  static std::once_flag flag;
  static bool is_rooted = false;

#if BAND_IS_MOBILE
  std::call_once(
      flag,
      [](bool& is_rooted) {
        is_rooted = RunCommand("su -c 'echo rooted'").find("rooted") !=
                    std::string::npos;
        BAND_LOG(LogSeverity::kInfo, "Is rooted: %d", is_rooted);
      },
      is_rooted);
#endif

  return is_rooted;
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