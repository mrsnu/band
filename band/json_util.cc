#include "band/json_util.h"

#include <sys/stat.h>

#include <fstream>

#include "band/logger.h"

namespace Band {

inline bool FileExists(const std::string& name) {
  struct stat buffer;
  return stat(name.c_str(), &buffer) == 0;
}

Json::Value LoadJsonObjectFromFile(std::string file_path) {
  Json::Value json_object;
  if (FileExists(file_path)) {
    std::ifstream in(file_path, std::ifstream::binary);
    in >> json_object;
  } else {
    BAND_LOG_PROD(BAND_LOG_WARNING, "There is no such file %s",
                  file_path.c_str());
  }
  return json_object;
}

void WriteJsonObjectToFile(const Json::Value& json_object,
                           std::string file_path) {
  std::ofstream out_file(file_path, std::ios::out);
  if (out_file.is_open()) {
    out_file << json_object;
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Cannot save profiled results to  %s",
                  file_path.c_str());
  }
}
}  // namespace Band