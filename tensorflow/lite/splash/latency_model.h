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
  ILatencyModel(worker_id_t wid) : wid_(wid) {}

  // init model parameters with default values
  virtual TfLiteStatus Init() = 0;

  // Get an estimation value of future latency 
  // after executing inference of the input model
  virtual int64_t Predict(int32_t model_id) = 0;

  // Update model parameters with the real latency 
  virtual TfLiteStatus Update(int32_t model_id, int64_t latency) = 0;

  worker_id_t GetWorkerId() {
    return wid_;
  }

 protected:
  worker_id_t wid_;

  double smoothing_factor_ = 0.01;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_LATENCY_MODEL_H_