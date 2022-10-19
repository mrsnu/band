#ifndef TENSORFLOW_LITE_MODEL_MANAGER_H_
#define TENSORFLOW_LITE_MODEL_MANAGER_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/thermal_model.h"
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
  TfLiteStatus Init();

  // get a list of workers which will not be throttled after the inference
  std::vector<worker_id_t> GetPossibleWorkers(Subgraph* subgraph);

  std::vector<thermal_t> GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph);

  // TEMP : for data collection
  int64_t GetFlops(const Subgraph* subgraph);
  int64_t GetMembytes(const Subgraph* subgraph);

  // Update model parameters with the prediction error
  TfLiteStatus Update(Job& job);

 private:
  std::vector<std::unique_ptr<IThermalModel>> thermal_models_;
  ResourceMonitor& resource_monitor_;

  std::unique_ptr<IThermalModel> BuildThermalModel(worker_id_t wid);

};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_MODEL_MANAGER_H_