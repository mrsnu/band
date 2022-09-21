#ifndef TENSORFLOW_LITE_LATENCY_MODEL_H_
#define TENSORFLOW_LITE_LATENCY_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"

namespace tflite {
namespace impl {

// Construct a prediction model for latency corresponding to 
// a target model of inference request, and provides the prediction
// value to schedulers. 
class LatencyModel {
 public:
  std::vector<int32_t> GetExpectedLatency(SubgraphKey key);

 private:
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_LATENCY_MODEL_H_