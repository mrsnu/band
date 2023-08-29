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

#include "band/json_util.h"

#include <sys/stat.h>

#include <fstream>

#include "band/logger.h"

#include "absl/strings/str_format.h"

namespace band {
namespace json {

inline bool FileExists(const std::string& name) {
  struct stat buffer;
  return stat(name.c_str(), &buffer) == 0;
}

inline bool IsEmpty(std::ifstream& ifs) {
  return ifs.peek() == std::ifstream::traits_type::eof();
}

Json::Value LoadFromFile(std::string file_path) {
  if (!FileExists(file_path)) {
    BAND_LOG_PROD(BAND_LOG_WARNING, "There is no such file %s",
                  file_path.c_str());
    return {};
  }

  std::ifstream in(file_path, std::ifstream::binary);
  if (IsEmpty(in)) {
    BAND_LOG_PROD(BAND_LOG_WARNING, "File %s is empty", file_path.c_str());
    return {};
  }

  Json::Value json_object;
  in >> json_object;
  return json_object;
}

absl::Status WriteToFile(const Json::Value& json_object,
                         std::string file_path) {
  std::ofstream out_file(file_path, std::ios::out);
  if (!out_file.is_open()) {
    return absl::InternalError(absl::StrFormat(
        "Cannot save profiled results to  %s", file_path.c_str()));
  }

  out_file << json_object;
  return absl::OkStatus();
}

bool Validate(const Json::Value& root, std::vector<std::string> required) {
  if (root.isNull()) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Please validate the json config file");
    return false;
  }

  for (auto key : required) {
    if (root[key].isNull()) {
      BAND_LOG_PROD(
          BAND_LOG_ERROR,
          "Please check if the argument %s is given in the config file",
          key.c_str());
      return false;
    }
  }

  return true;
}

}  // namespace json
}  // namespace band