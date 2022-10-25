#ifndef TENSORFLOW_LITE_CLOUD_LATENCY_MODEL_H_
#define TENSORFLOW_LITE_CLOUD_LATENCY_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/latency_model.h"

namespace tflite {
namespace impl {

class CloudLatencyModel : public ILatencyModel {
 public:
  CloudLatencyModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : ILatencyModel(wid, resource_monitor) {}

  TfLiteStatus Init() override;

  int64_t Predict(int32_t model_id) override;
  int64_t PredictThrottled(int32_t model_id) override;

  TfLiteStatus Update(int32_t model_id, int64_t latency) override;

 private:
  std::unordered_map<int, int64_t> model_latency_table_; // {model_id, latency}

  int64_t EstimateInputSize(const Subgraph* subgraph);
  int64_t EstimateOutputSize(const Subgraph* subgraph);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_CLOUD_LATENCY_MODEL_H_