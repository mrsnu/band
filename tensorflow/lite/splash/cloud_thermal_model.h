#ifndef TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/thermal_model.h"

namespace tflite {
namespace impl {

class CloudThermalModel : public IThermalModel {
 public:

  TfLiteStatus Init() override;

  std::unordered_map<worker_id_t, ThermalInfo> Predict(SubgraphKey key) override;

  TfLiteStatus Update(std::vector<ThermalInfo> error) override;

 private:
  // Linear regressor
  std::vector<int32_t> temperature;
  std::int64_t input_size;
  std::int64_t output_size;
  std::int64_t rssi;
  std::int64_t waiting_time;
  
  // Model parameter
  std::vector<std::vector<double>> temp_param;
  std::vector<double> input_param;
  std::vector<double> output_param;
  std::vector<double> rssi_param;
  std::vector<double> waiting_param;
  std::vector<double> error_param;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_