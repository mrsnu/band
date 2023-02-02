#ifndef BAND_JSON_UTIL_H_
#define BAND_JSON_UTIL_H_

#include <json/json.h>

#include <type_traits>
#include <typeinfo>

#include "band/common.h"
#include "band/logger.h"

namespace Band {
namespace json {
// load data from the given file
// if there is no such file, then the json object will be empty
Json::Value LoadFromFile(std::string file_path);
// write json object
BandStatus WriteToFile(const Json::Value& json_object, std::string file_path);
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
}  // namespace Band

#endif