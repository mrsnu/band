#ifndef TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/thermal_model.h"

namespace tflite {
namespace impl {

class ProcessorThermalModel : IThermalModel {
 public:
  TfLiteStatus Init() override;
  std::unordered_map<worker_id_t, ThermalInfo> GetFutureTemperature(SubgraphKey key) override;
  TfLiteStatus Update(std::vector<ThermalInfo> error) override;
  
 private:
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_