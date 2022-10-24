#ifndef TENSORFLOW_LITE_PROCESSOR_LATENCY_MODEL_H_
#define TENSORFLOW_LITE_PROCESSOR_LATENCY_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/latency_model.h"

namespace tflite {
namespace impl {

class ProcessorLatencyModel : public ILatencyModel {
 public:
  ProcessorLatencyModel(worker_id_t wid) : ILatencyModel(wid) {}

  TfLiteStatus Init() override;

  int64_t Predict(int32_t model_id) override;
  int64_t PredictThrottled(int32_t model_id) override;

  TfLiteStatus Update(int32_t model_id, int64_t latency) override;
 
 private:
  std::unordered_map<int, int64_t> model_latency_table_; // {model_id, latency}
  std::unordered_map<int, int64_t> model_throttled_latency_table_; // {model_id, latency}
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_LATENCY_MODEL_H_