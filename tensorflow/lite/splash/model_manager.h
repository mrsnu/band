#ifndef TENSORFLOW_LITE_MODEL_MANAGER_H_
#define TENSORFLOW_LITE_MODEL_MANAGER_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/thermal_model.h"
#include "tensorflow/lite/splash/latency_model.h"
#include "tensorflow/lite/splash/resource_monitor.h"

namespace tflite {
namespace impl {

class ResourceMonitor;
class CommonThermalModel;

// Construct a prediction model for heat generation corresponding to 
// a target model of inference request, and provides the prediction
// value to schedulers.
class ModelManager {
 public:
  ModelManager(ResourceMonitor& resource_monitor) : resource_monitor_(resource_monitor) {}

  ~ModelManager();

  // Initialize model parameters with default values
  TfLiteStatus Init(ResourceConfig& config, bool is_thermal_aware);

  // Check if it's throttled or will occur thermal throttling
  bool IsAvailableWorker(worker_id_t wid, Subgraph* subgraph);

  thermal_t GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph);
  int64_t GetPredictedLatency(worker_id_t wid, Subgraph* subgraph);
  // int64_t GetPredictedThrottledLatency(worker_id_t wid, int32_t model_id);
  std::pair<thermal_t, int64_t>
  GetPredictedTempAndLatency(worker_id_t wid, Subgraph* subgraph);

  // Update model parameters with the prediction error
  TfLiteStatus Update(Job& job, Subgraph* subgraph);

  TfLiteStatus ProfileLatency(Subgraph* subgraph, int64_t latency);

  TfLiteStatus Close();

 private:
  std::vector<std::unique_ptr<IThermalModel>> thermal_models_;
  std::vector<std::unique_ptr<ILatencyModel>> latency_models_;
  ResourceMonitor& resource_monitor_;

  std::unique_ptr<IThermalModel> BuildThermalModel(worker_id_t wid);
  std::unique_ptr<ILatencyModel> BuildLatencyModel(worker_id_t wid, bool is_thermal_aware);

};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_MODEL_MANAGER_H_