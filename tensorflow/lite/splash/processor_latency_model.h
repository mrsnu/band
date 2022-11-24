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
  ProcessorLatencyModel(worker_id_t wid, ResourceMonitor& resource_monitor, bool is_thermal_aware)
  : ILatencyModel(wid, resource_monitor, is_thermal_aware) {}

  TfLiteStatus Init(ResourceConfig& config) override;

  int64_t Predict(Subgraph* subgraph) override;

  TfLiteStatus Update(Job job, Subgraph* subgraph) override;

  TfLiteStatus Profile(int32_t model_id, int64_t latency) override;

  TfLiteStatus Close() override;
 
 private:

  std::unordered_map<int, std::unordered_map<int, int64_t>> model_latency_table_; // {model_id, {temp, latency}}

  std::unordered_map<int, int> minimum_profiled_count_; // {model_id, count}
  int minimum_profiled_threshold_ = 3;
  std::string model_path_;

  int64_t FindNearestValue(int model_id, thermal_t target_temp);
  void LoadModelParameter(string latency_model_path);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_LATENCY_MODEL_H_