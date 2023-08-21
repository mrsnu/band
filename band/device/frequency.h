#ifndef BAND_DEVICE_FREQUENCY_H_
#define BAND_DEVICE_FREQUENCY_H_

#include <string>
#include <map>

#include "absl/status/status.h"

namespace band {

class DeviceConfig;
enum class DeviceFlag : size_t;

using FreqMap = std::map<DeviceFlag, double>;

class Frequency {
 public:
  explicit Frequency(DeviceConfig config);
  double GetFrequency(DeviceFlag device_flag);
  FreqMap GetAllFrequency();

 private:
  bool CheckFrequency(std::string path);
  std::map<DeviceFlag, std::string> freq_device_map_;
};

}  // namespace band

#endif  // BAND_DEVICE_FREQUENCY_H_