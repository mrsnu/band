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

  TfLiteStatus Init(int32_t window_size) override;

  thermal_t Predict(const Subgraph* subgraph, 
                    const int64_t latency, 
                    std::vector<thermal_t> current_temp) override;

  thermal_t PredictTarget(const Subgraph* subgraph, 
                    const int64_t latency, 
                    std::vector<thermal_t> current_temp) override;

  TfLiteStatus Update(Job job) override;

 private:
  int32_t window_size_;
  
  // Model parameter
  std::vector<std::vector<double>> temp_param_;
  std::vector<double> input_param_;
  std::vector<double> output_param_;
  std::vector<double> rssi_param_;
  std::vector<double> waiting_param_;
  std::vector<double> error_param_;

  int64_t EstimateInputSize(const Subgraph* subgraph);
  int64_t EstimateOutputSize(const Subgraph* subgraph);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_