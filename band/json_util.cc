#include "band/json_util.h"

#include <sys/stat.h>

#include <fstream>

#include "band/logger.h"

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