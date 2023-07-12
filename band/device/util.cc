#include "band/device/util.h"

#include <fstream>
#include <mutex>
#include <sstream>

#include "band/logger.h"
#include "util.h"

namespace band {
namespace device {
template <typename T>
absl::StatusOr<T> TryRead(std::vector<std::string> paths) {
  for (const std::string& path : paths) {
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      T output;
      fs >> output;
      return output;
    }
  }
  return absl::NotFoundError("No available path");
}

absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths) {
  return TryRead<size_t>(paths);
}

absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths) {
  for (const std::string& path : paths) {
    std::fstream fs(path, std::fstream::in);
    if (fs.is_open()) {
      std::vector<size_t> outputs;
      size_t output;
      while (fs >> output) {
        outputs.push_back(output);
      }
      return outputs;
    }
  }
  return absl::NotFoundError("No available path");
}

absl::StatusOr<std::string> TryReadString(std::vector<std::string> paths) {
  return TryRead<std::string>(paths);
}

bool SupportsDevice() {
#if BAND_SUPPORT_DEVICE
  return true;
#else
  return false;
#endif
}

bool IsRooted() {
  static std::once_flag flag;
  static bool is_rooted = false;

#if BAND_SUPPORT_DEVICE
  std::call_once(
      flag,
      [](bool& is_rooted) {
        std::string command = "su -c 'echo Rooted'";
        std::string result = "";

        FILE* pipe = popen(command.c_str(), "r");
        if (pipe != nullptr) {
          char buffer[128];
          while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
          }
          pclose(pipe);
        }

        is_rooted = result.find("Rooted") != std::string::npos;
        BAND_LOG_INTERNAL(BAND_LOG_INFO, "Is rooted: %d", is_rooted);
      },
      is_rooted);
#endif

  return is_rooted;
}
}  // namespace device
}  // namespace band