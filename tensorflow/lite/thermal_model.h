#ifndef TENSORFLOW_LITE_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/resource_monitor.h"

namespace tflite {
namespace impl {

// Construct a prediction model for heat generation corresponding to 
// a target model of inference request, and provides the prediction
// value to schedulers.
class ThermalModel {
 public:
  std::vector<ThermalInfo> GetHeatGeneration(SubgraphKey key);

 private:
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_THERMAL_MODEL_H_