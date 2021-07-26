#include "tensorflow/lite/processors/gpu.h"
#include "tensorflow/lite/processors/util.h"

#include <cstring>
#if defined __ANDROID__ || defined __linux__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tflite {
namespace impl {

std::vector<std::string> GetPaths(std::string suffix) {
  std::vector<std::string> device_paths = {
    "/sys/class/kgsl/kgsl-3d0/" // Pixel4
  };
  for (size_t i = 0; i < device_paths.size(); i++) {
    device_paths[i] += suffix;
  }
  return device_paths;
}

int GetGPUMinFrequencyKhz() {
  return TryReadInt(GetPaths("min_clock_mhz")) * 1000;
}

int GetGPUMaxFrequencyKhz() {
  return TryReadInt(GetPaths("max_clock_mhz")) * 1000;
}

int GetGPUFrequencyKhz() {
  return TryReadInt(GetPaths("clock_mhz")) * 1000;
}

std::vector<int> GetGPUAvailableFrequenciesKhz() {
  std::vector<int> frequenciesMhz = TryReadInts(GetPaths("freq_table_mhz"));
  for (size_t i = 0; i < frequenciesMhz.size(); i++) {
    frequenciesMhz[i] *= 1000;
  }
  return frequenciesMhz;
}

std::vector<std::pair<int, int>> GetGPUClockStats() {
  std::vector<int> frequencies = GetGPUAvailableFrequenciesKhz();
  std::vector<int> clock_stats = TryReadInts(GetPaths("gpu_clock_stats"));

  std::vector<std::pair<int, int>> frequency_stats(frequencies.size());
  for (size_t i = 0; i < frequency_stats.size(); i++) {
    frequency_stats[i] = {frequencies[i], clock_stats[i]};
  }

  return frequency_stats;
}
}  // namespace impl
}  // namespace tflite
