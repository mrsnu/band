#include "band/config.h"

namespace band {
namespace test {

static DeviceConfig GetPixel4DeviceConfig() {
  static DeviceConfig pixel4_config;
  pixel4_config.cpu_therm_index = 47;
  pixel4_config.gpu_therm_index = 32;
  pixel4_config.dsp_therm_index = 52;
  pixel4_config.npu_therm_index = 38;
  pixel4_config.target_therm_index = 74;
  
  pixel4_config.runtime_freq_path = "/sys/devices/system/cpu/cpufreq/policy4";
  pixel4_config.cpu_freq_path = "/sys/devices/system/cpu/cpufreq/policy7";
  pixel4_config.gpu_freq_path = "/sys/class/devfreq/2c00000.qcom,kgsl-3d0";
  pixel4_config.dsp_freq_path = "";
  pixel4_config.npu_freq_path = "";

  pixel4_config.latency_log_path = "/data/local/tmp/splash/latency.log";
  pixel4_config.therm_log_path = "/data/local/tmp/splash/therm.log";
  pixel4_config.freq_log_path = "/data/local/tmp/splash/freq.log";
  return pixel4_config;
}

// static DeviceConfig GetGalaxyS20DeviceConfig() {
//   static DeviceConfig galaxy_s20_config;
//   return galaxy_s20_config;
// }

}  // namespace test
}  // namespace band