#ifndef TENSORFLOW_LITE_THERMAL_MODEL_MANAGER_H_
#define TENSORFLOW_LITE_THERMAL_MODEL_MANAGER_H_

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
class ThermalModelManager {
 public:
  ThermalModelManager(ResourceMonitor& resource_monitor) : resource_monitor_(resource_monitor) {}

  ~ThermalModelManager();

  // Initialize model parameters with default values
  TfLiteStatus Init();

  // Get an estimation value of future temperature 
  // after executing inference of the input model
  std::vector<worker_id_t> GetPossibleWorkers(Subgraph* subgraph);

  std::vector<thermal_t> GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph);

  // TEMP : for data collection
  int64_t GetFlops(const Subgraph* subgraph);
  int64_t GetMembytes(const Subgraph* subgraph);

  // Update model parameters with the prediction error
  TfLiteStatus Update(Job& job);

 private:
  std::vector<std::unique_ptr<IThermalModel>> models_;
  ResourceMonitor& resource_monitor_;

  std::unique_ptr<IThermalModel> BuildModel(worker_id_t wid);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_THERMAL_MODEL_MANAGER_H_