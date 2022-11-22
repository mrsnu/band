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
 public:
  ProcessorLatencyModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : ILatencyModel(wid, resource_monitor) {}

  TfLiteStatus Init() override;

  int64_t Predict(Subgraph* subgraph) override;

  TfLiteStatus Update(Job job, Subgraph* subgraph) override;

  TfLiteStatus Profile(int32_t model_id, int64_t latency) override;
 
 private:
  std::unordered_map<int, std::unordered_map<int, int64_t>> model_latency_table_; // {model_id, {temp, latency}}

  std::unordered_map<int, std::unordered_map<int, int>> minimum_profiled_count_; // {model_id, {temp, count}}
  int minimum_profiled_threshold_ = 3;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_LATENCY_MODEL_H_