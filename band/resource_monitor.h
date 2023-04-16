#ifndef BAND_RESOURCE_MONITOR_H_
#define BAND_RESOURCE_MONITOR_H_

#include <map>
#include <tuple>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace band {

using ThermalStatus = std::map<std::string, int32_t>;
using Frequency = std::map<std::string, int32_t>;
using NetworkStatus = std::map<std::string, int32_t>;

class ResourceMonitor {
 public:
  static absl::StatusOr<ResourceMonitor> Create();
  absl::Status Init();

  absl::StatusOr<const ThermalStatus> GetCurrentThermalStatus() const;
  absl::StatusOr<const Frequency> GetCurrentFrequency() const;
  absl::StatusOr<const NetworkStatus> GetCurrentNetworkStatus() const;
  std::tuple<const ThermalStatus, const Frequency, const NetworkStatus>
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

  std::vector<std::string> tzs_;
  std::vector<std::string> cpufreqs_;
  std::vector<std::string> devfreqs_;
};

}  // namespace band

#endif  // BAND_RESOURCE_MONITOR_H_