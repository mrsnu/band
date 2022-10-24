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
  auto it = model_latency_table_.find(model_id);
  if (it != model_latency_table_.end()) {
    bool throttled = false;
    // TODO: Thorttling detection
    if (!throttled) {
      int64_t prev_latency = model_latency_table_[model_id];
      model_latency_table_[model_id] =
          smoothing_factor_ * latency +
          (1 - smoothing_factor_) * prev_latency;
    } else {
      // TODO: Update threshold value on resource_monitor
      int64_t prev_latency = model_throttled_latency_table_[model_id];
      model_throttled_latency_table_[model_id] =
          smoothing_factor_ * latency +
          (1 - smoothing_factor_) * prev_latency;
    }
    return kTfLiteOk;
  } else {
    model_latency_table_[model_id] = latency;
    return kTfLiteOk;
  }
}

} // namespace impl
} // namespace tflite