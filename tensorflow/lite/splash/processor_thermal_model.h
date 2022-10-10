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

class ProcessorThermalModel : public IThermalModel {
 public:

  TfLiteStatus Init() override;

  std::unordered_map<worker_id_t, ThermalInfo> Predict(SubgraphKey key) override;

  TfLiteStatus Update(std::vector<ThermalInfo> error) override;
  
 private:
  // Linear regressor
  std::vector<int32_t> temperature; // Get from resource monitor
  std::vector<int32_t> frequency;
  std::int64_t flops;
  std::int64_t membytes;
  
  // Model parameter
  std::vector<std::vector<double>> temp_param;
  std::vector<std::vector<double>> freq_param;
  std::vector<double> flops_param;
  std::vector<double> membytes_param;
  std::vector<double> error_param;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_