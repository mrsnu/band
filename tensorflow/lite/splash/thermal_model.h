#ifndef TENSORFLOW_LITE_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/splash/resource_monitor.h"

namespace tflite {
namespace impl {

class IThermalModel {
 public:
  // init model parameters with default values
  virtual TfLiteStatus Init();

  // Get an estimation value of future temperature 
  // after executing inference of the input model
  virtual std::unordered_map<worker_id_t, ThermalInfo> Predict(SubgraphKey key);

  // Update model parameters with the prediction error
  virtual TfLiteStatus Update(std::vector<ThermalInfo> error);

 private:

};

// Construct a prediction model for heat generation corresponding to 
// a target model of inference request, and provides the prediction
// value to schedulers.
class ThermalModel {
 public:
  ThermalModel(ResourceMonitor& resource_monitor) : resource_monitor_(resource_monitor) {}

  ~ThermalModel();

  // Initialize model parameters with default values
  TfLiteStatus Init();

  // Get an estimation value of future temperature 
  // after executing inference of the input model
  std::vector<worker_id_t> GetPossibleWorkers(SubgraphKey key);

  // Update model parameters with the prediction error
  TfLiteStatus Update(std::vector<ThermalInfo> error, worker_id_t wid);

 private:
  std::unordered_map<worker_id_t, IThermalModel *> models_;
  ResourceMonitor& resource_monitor_;
  
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_THERMAL_MODEL_H_