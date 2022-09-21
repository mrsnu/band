#include "tensorflow/lite/thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

std::vector<ThermalInfo> ThermalModel::GetHeatGeneration(SubgraphKey key) {
    return std::vector<ThermalInfo>();
}

} // namespace impl
} // namespace tflite