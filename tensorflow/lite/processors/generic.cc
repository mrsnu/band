#include "tensorflow/lite/processors/generic.h"
#include "tensorflow/lite/processors/util.h"

#include <cstring>
#if defined __ANDROID__ || defined __linux__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace tflite {
namespace impl {
namespace generic {

std::vector<std::string> GetPaths(TfLiteDeviceFlags device_flag, std::string suffix) {
  std::vector<std::string> device_paths;
#if defined __ANDROID__ || defined __linux__
  // TODO: Add more device-specific path
  if (device_flag == kTfLiteNPU) {
	device_paths = {
		"/sys/devices/platform/17000060.devfreq_npu/devfreq/17000060.devfreq_npu/"  // Galaxy S21
	};
  } else if (device_flag == kTfLiteDSP) {
	device_paths = {
	  "/sys/devices/platform/soc/soc:qcom,cdsp-cdsp-l3-lat/devfreq/soc:qcom,cdsp-cdsp-l3-lat/" // Pixel 4 Hexagon DSP
	};
  }
#endif
  for (size_t i = 0; i < device_paths.size(); i++) {
    device_paths[i] += suffix;
  }
  return device_paths;
}


int GetMinFrequencyKhz(TfLiteDeviceFlags device_flag) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt(GetPaths(device_flag, "min_freq")) * 1000;
#elif
  return -1;
#endif
}

int GetMaxFrequencyKhz(TfLiteDeviceFlags device_flag) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt(GetPaths(device_flag, "max_freq")) * 1000;
#elif
  return -1;
#endif
}

int GetFrequencyKhz(TfLiteDeviceFlags device_flag) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt(GetPaths(device_flag, "cur_freq")) * 1000;
#elif
  return -1;
#endif
}

int GetTargetFrequencyKhz(TfLiteDeviceFlags device_flag) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt(GetPaths(device_flag, "target_freq")) * 1000;
#elif
  return -1;
#endif
}

int GetPollingIntervalMs(TfLiteDeviceFlags device_flag) {
#if defined __ANDROID__ || defined __linux__
  return TryReadInt(GetPaths(device_flag, "polling_interval"));
#elif
  return -1;
#endif
}

std::vector<int> GetAvailableFrequenciesKhz(TfLiteDeviceFlags device_flag) {
  std::vector<int> frequenciesMhz;
#if defined __ANDROID__ || defined __linux__
  frequenciesMhz = TryReadInts(GetPaths(device_flag, "available_frequencies"));
  for (size_t i = 0; i < frequenciesMhz.size(); i++) {
    frequenciesMhz[i] *= 1000;
  }
#endif
  return frequenciesMhz;
}

std::vector<std::pair<int, int>> GetClockStats(TfLiteDeviceFlags device_flag) {
  std::vector<std::pair<int, int>> frequency_stats;

#if defined __ANDROID__ || defined __linux__
  std::vector<int> frequencies = GetAvailableFrequenciesKhz(device_flag);
  std::vector<int> clock_stats = TryReadInts(GetPaths(device_flag, "time_in_state"));

  frequency_stats.resize(frequencies.size());
  for (size_t i = 0; i < frequency_stats.size(); i++) {
    frequency_stats[i] = {frequencies[i], clock_stats[i]};
  }
#endif
  return frequency_stats;
}

} // namespace generic
}  // namespace impl
}  // namespace tflite
