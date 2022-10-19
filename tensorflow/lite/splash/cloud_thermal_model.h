#ifndef TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/thermal_model.h"

namespace tflite {
namespace impl {

class CloudThermalModel : public IThermalModel {
 public:
  CloudThermalModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : IThermalModel(wid, resource_monitor) {}

  TfLiteStatus Init(int32_t worker_size, int32_t window_size) override;

  std::vector<thermal_t> Predict(const Subgraph* subgraph) override;

  TfLiteStatus Update(Job job) override;

 private:
  // Linear regressor
  std::vector<thermal_t> temperature;
  std::int64_t input_size;
  std::int64_t output_size;
  std::int64_t rssi;
  std::int64_t waiting_time;

  int32_t window_size_;
  
  // Model parameter
  std::vector<std::vector<double>> temp_param_;
  std::vector<double> input_param_;
  std::vector<double> output_param_;
  std::vector<double> rssi_param_;
  std::vector<double> waiting_param_;
  std::vector<double> error_param_;

  std::vector<thermal_t> EstimateFutureTemperature(const std::vector<thermal_t> temp,
                                                   const int64_t input_size,
                                                   const int64_t output_size,
                                                   const int64_t rssi,
                                                   const int64_t waiting_time);
  int64_t EstimateInputSize(const Subgraph* subgraph);
  int64_t EstimateOutputSize(const Subgraph* subgraph);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_