#ifndef TENSORFLOW_LITE_LATENCY_MODEL_H_
#define TENSORFLOW_LITE_LATENCY_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/splash/resource_monitor.h"

namespace tflite {
namespace impl {

class Subgraph;

class ILatencyModel {
 public:
  ILatencyModel(worker_id_t wid, ResourceMonitor& resource_monitor) 
    : wid_(wid), resource_monitor_(resource_monitor) {}

  // init model parameters with default values
  virtual TfLiteStatus Init() = 0;

  // Get an estimation value of future latency 
  // after executing inference of the input model
  virtual int64_t Predict(int32_t model_id) = 0;

  // Get an estimation value of future latency 
  // after executing inference of the input model
  virtual int64_t PredictThrottled(int32_t model_id) = 0;

  // Update model parameters with the real latency 
  virtual TfLiteStatus Update(int32_t model_id, int64_t latency) = 0;

  virtual TfLiteStatus Profile(int32_t model_id, int64_t latency) = 0;

  worker_id_t GetWorkerId() {
    return wid_;
  }

  ResourceMonitor& GetResourceMonitor() {
    return resource_monitor_;
  }

 protected:
  worker_id_t wid_;
  ResourceMonitor& resource_monitor_;

  double smoothing_factor_ = 0.1;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_LATENCY_MODEL_H_