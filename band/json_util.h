#ifndef BAND_JSON_UTIL_H_
#define BAND_JSON_UTIL_H_

#include <json/json.h>

#include "band/common.h"

namespace Band {

// load data from the given file
// if there is no such file, then the json object will be empty
Json::Value LoadJsonObjectFromFile(std::string file_path);

// Write json object.
BandStatus WriteJsonObjectToFile(const Json::Value& json_object,
                                 std::string file_path);

}  // namespace Band

#endif