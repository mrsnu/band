#include "tensorflow/lite/splash/processor_latency_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "thermal", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus ProcessorLatencyModel::Init() {
  // Do nothing
  return kTfLiteOk;
}

int64_t ProcessorLatencyModel::Predict(int32_t model_id) {
  auto it = model_latency_table_.find(model_id);
  if (it != model_latency_table_.end()) {
    return it->second;
  } else {
    return 0; // Minimum value to be selected
  }
}

int64_t ProcessorLatencyModel::PredictThrottled(int32_t model_id) {
  auto it = model_throttled_latency_table_.find(model_id);
  if (it != model_throttled_latency_table_.end()) {
    return it->second;
  } else {
    return 0; // Minimum value to be selected
  }
}

TfLiteStatus ProcessorLatencyModel::Update(int32_t model_id, int64_t latency) {
  thermal_t current_temp = GetResourceMonitor().GetTemperature(wid_);
  thermal_t threshold = GetResourceMonitor().GetThrottlingThreshold(wid_);
  if (current_temp > threshold) {
    UpdateThrottledLatency(model_id, latency); 
    return kTfLiteOk;
  }

  auto it = model_latency_table_.find(model_id);
  if (it != model_latency_table_.end()) {
    if (IsThrottled(model_id, latency, current_temp)) { // If new throttling threshold detected
      LOGI("PLM::Update Newly Throttling detected in worker[%d] on current temp = %d", wid_, current_temp);
      GetResourceMonitor().SetThrottlingThreshold(wid_, current_temp);
      UpdateThrottledLatency(model_id, latency);
    } else {
      int64_t prev_latency = model_latency_table_[model_id];
      model_latency_table_[model_id] =
          smoothing_factor_ * latency +
          (1 - smoothing_factor_) * prev_latency;
    }
  } else {
    model_latency_table_[model_id] = latency;
  }
  return kTfLiteOk;
}

bool ProcessorLatencyModel::IsThrottled(int32_t model_id, int64_t latency, thermal_t current_temp) {
  auto it = model_throttled_latency_table_.find(model_id);
  if (it != model_throttled_latency_table_.end()) { 
    // If table has previous throttled_latency value,
    // the system must have experienced this temp before.
    return false;
  }
  int64_t prev_latency = model_latency_table_[model_id];
  if (latency - prev_latency > prev_latency * throttled_diff_rate_ && current_temp > throttled_temp_min_) {
    return true;
  }
}

TfLiteStatus ProcessorLatencyModel::UpdateThrottledLatency(int32_t model_id, int64_t latency) {
  auto it = model_throttled_latency_table_.find(model_id);
  if (it != model_throttled_latency_table_.end()) { 
    int64_t prev_latency = model_throttled_latency_table_[model_id];
    model_throttled_latency_table_[model_id] =
        smoothing_factor_ * latency +
        (1 - smoothing_factor_) * prev_latency;
  } else {
    model_throttled_latency_table_[model_id] = latency;
  }
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite