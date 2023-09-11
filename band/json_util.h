/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_JSON_UTIL_H_
#define BAND_JSON_UTIL_H_

#include <type_traits>
#include <typeinfo>

#include "band/common.h"
#include "band/logger.h"

#include "json/json.h"
#include "absl/status/status.h"

namespace band {
namespace json {
// load data from the given file
// if there is no such file, then the json object will be empty
Json::Value LoadFromFile(std::string file_path);
// write json object
absl::Status WriteToFile(const Json::Value& json_object, std::string file_path);
// validate the root, returns true if root is valid and has all required fields
bool Validate(const Json::Value& root, std::vector<std::string> required);
template <typename T>
bool AssignIfValid(T& lhs, const Json::Value& value, const char* key) {
  if (!value[key].isNull()) {
    lhs = value[key].as<T>();
    return true;
  } else {
    return false;
  }
}
}  // namespace json
}  // namespace band

#endif