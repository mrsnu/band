#ifndef BAND_DEVICE_FREQUENCY_H_
#define BAND_DEVICE_FREQUENCY_H_

#include <map>
#include <string>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"

namespace band {

using FreqMap = std::map<FreqFlag, double>;

class Frequency {
 public:
  explicit Frequency(DeviceConfig config);
  double GetFrequency(FreqFlag device_flag);

  absl::Status SetRuntimeFrequency(double freq);
  absl::Status SetCpuFrequency(double freq);
  absl::Status SetGpuFrequency(double freq);

  FreqMap GetAllFrequency();

  std::map<FreqFlag, std::vector<double>> GetAllAvailableFrequency();

 private:
  DeviceConfig config_;
  bool CheckFrequency(std::string path);
  absl::Status SetFrequencyWithPath(const std::string& path, double freq,
                                    size_t multiplier);

  std::map<FreqFlag, std::string> freq_device_map_;
  std::map<FreqFlag, std::vector<double>> freq_available_map_;
  std::vector<double> freq_runtime_available_;

  float cpu_freq_multiplier = 1.0E-6f;
  float dev_freq_multiplier = 1.0E-9f;
  size_t cpu_freq_multiplier_w = 1.0E+6f;
  size_t dev_freq_multiplier_w = 1.0E+9f;
  const std::map<size_t, size_t> gpu_freq_map_ = {{585000000, 0},
                                                  {499200000, 1},
                                                  {427000000, 2},
                                                  {345000000, 3},
                                                  {257000000, 4}};
};

}  // namespace band

#endif  // BAND_DEVICE_FREQUENCY_H_