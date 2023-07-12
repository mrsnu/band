#ifndef BAND_RESOURCE_MONITOR_H_
#define BAND_RESOURCE_MONITOR_H_

#include <fstream>
#include <map>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#define DEFINE_TO_JSON                                          \
  std::string ToJson() const {                                  \
    std::string ret = "{";                                      \
    for (auto& pair : status) {                                 \
      auto& key = pair.first;                                   \
      auto& value = pair.second;                                \
      ret += "\"" + key + "\": " + std::to_string(value) + ","; \
    }                                                           \
    ret.pop_back();                                             \
    ret += "}";                                                 \
    return ret;                                                 \
  }

namespace band {

struct Thermal {
  std::map<std::string, int32_t> status;
  DEFINE_TO_JSON;
};

struct Frequency {
  std::map<std::string, int32_t> status;
  DEFINE_TO_JSON;
};

struct Network {
  std::map<std::string, int32_t> status;
  DEFINE_TO_JSON;
};

class ResourceMonitor {
 public:
  ~ResourceMonitor();
  static ResourceMonitor& Create(std::string log_path = "");
  absl::Status Init(std::string log_path);

  absl::StatusOr<const Thermal> GetCurrentThermal() const;
  absl::StatusOr<const Frequency> GetCurrentFrequency() const;
  absl::StatusOr<const Network> GetCurrentNetwork() const;
  absl::StatusOr<std::tuple<const Thermal, const Frequency, const Network>>
  GetCurrentStatus() const;

  absl::Status AddTemperatureResource(std::string resource_path);
  absl::Status AddCPUFrequencyResource(std::string resource_path);
  absl::Status AddDevFrequencyResource(std::string resource_path);
  absl::Status AddNetworkResource(std::string resource_path);

  std::vector<std::string> GetDetectedThermalZonePaths() const;
  std::vector<std::string> GetDetectedCpuFreqPaths() const;
  std::vector<std::string> GetDetectedDevFreqPaths() const;
  std::vector<std::string> GetDetectedNetworkPaths() const;

 private:
  ResourceMonitor() = default;
  std::vector<std::string> GetThermalZonePaths() const;
  std::vector<std::string> GetCpuFreqPaths() const;
  std::vector<std::string> GetDevFreqPaths() const;
  std::vector<std::string> GetNetworkPaths() const;

  const char* THERMAL_ZONE_BASE_PATH = "/sys/class/thermal/";
  const char* CPUFREQ_BASE_PATH = "/sys/devices/system/cpu/cpufreq/";
  const char* DEVFREQ_BASE_PATH = "/sys/class/devfreq/";

  mutable std::ofstream log_file_;

  std::vector<std::string> tzs_;
  std::vector<std::string> cpufreqs_;
  std::vector<std::string> devfreqs_;
};

}  // namespace band

#endif  // BAND_RESOURCE_MONITOR_H_