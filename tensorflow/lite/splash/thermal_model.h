#ifndef TENSORFLOW_LITE_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/splash/resource_monitor.h"

namespace tflite {
namespace impl {

class Subgraph;

class IThermalModel {
 public:
  IThermalModel(worker_id_t wid, 
                         ResourceMonitor& resource_monitor)
                         : wid_(wid), resource_monitor_(resource_monitor) {}

  // init model parameters with default values
  virtual TfLiteStatus Init(ResourceConfig& config) = 0;

  // Get an estimation value of future temperature 
  // after executing inference of the input model
  virtual thermal_t Predict(const Subgraph* subgraph, 
                            const int64_t latency, 
                            std::vector<thermal_t> current_temp) = 0;

  // Get an estimation value of target future temperature 
  // after executing inference of the input model
  virtual thermal_t PredictTarget(const Subgraph* subgraph, 
                                  const int64_t latency, 
                                  std::vector<thermal_t> current_temp) = 0;

  // Update model parameters with the prediction error
  virtual TfLiteStatus Update(Job job, const Subgraph* subgraph) = 0;

  worker_id_t GetWorkerId() {
    return wid_;
  }

  ResourceMonitor& GetResourceMonitor() {
    return resource_monitor_;
  }

 protected:
  worker_id_t wid_;
  ResourceMonitor& resource_monitor_;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_THERMAL_MODEL_H_