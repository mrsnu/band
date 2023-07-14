#include "band/device/util.h"

#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#include "band/logger.h"
#include "util.h"

namespace band {
namespace device {

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

bool IsRooted() {
  static std::once_flag flag;
  static bool is_rooted = false;

#if BAND_IS_MOBILE
  std::call_once(
      flag,
      [](bool& is_rooted) {
        is_rooted = RunCommand("su -c 'echo rooted'").find("rooted") !=
                    std::string::npos;
        BAND_LOG_INTERNAL(BAND_LOG_INFO, "Is rooted: %d", is_rooted);
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

std::map<DeviceFlag, std::string> GetDevfreqPaths() {
  std::string model = GetDeviceProperty("ro.product.model").value_or("");
  // galaxy s20 (todo: fix this)
  if (model == "SM-981N") {
    return {
        {DeviceFlag::kDSP, "/sys/class/devfreq/exynos5-busfreq-dmc/cur_freq"},
        {DeviceFlag::kGPU, "/sys/class/devfreq/exynos5-busfreq-mif/cur_freq"},
        {DeviceFlag::kNPU, "/sys/class/devfreq/exynos5-busfreq-mif/cur_freq"}};
  }
  // other devices

  return std::map<DeviceFlag, std::string>();
}

}  // namespace device
}  // namespace band