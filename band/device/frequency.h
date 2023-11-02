#ifndef BAND_DEVICE_FREQUENCY_H_
#define BAND_DEVICE_FREQUENCY_H_

#include <map>
#include <string>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"

namespace band {

using FreqMap = std::map<DeviceFlag, double>;

class Frequency {
 public:
  explicit Frequency(DeviceConfig config);
  double GetFrequency(DeviceFlag device_flag);
  double GetRuntimeFrequency();

  absl::Status SetFrequency(DeviceFlag device_flag, double freq);
  absl::Status SetRuntimeFrequency(double freq);

  FreqMap GetAllFrequency();

  std::map<DeviceFlag, std::vector<double>> GetAllAvailableFrequency();
  std::vector<double> GetRuntimeAvailableFrequency();

 private:
  DeviceConfig config_;
  bool CheckFrequency(std::string path);
  absl::Status SetCpuFrequency(double freq);
  absl::Status SetDevFrequency(DeviceFlag device_flag, double freq);
  absl::Status SetFrequencyWithPath(const std::string& path, double freq,
                                    size_t multiplier);

  std::string runtime_cpu_path_;
  std::map<DeviceFlag, std::string> freq_device_map_;
  std::map<DeviceFlag, std::vector<double>> freq_available_map_;
};

}  // namespace band

#endif  // BAND_DEVICE_FREQUENCY_H_