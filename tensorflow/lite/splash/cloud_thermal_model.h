#ifndef TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
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

  TfLiteStatus Update(Job job, const Subgraph* subgraph) override;

 private:
  // Log buffer
  Eigen::MatrixXd targetX;
  Eigen::VectorXd targetY;
  uint32_t log_size_ = 0;
  int window_size_;
  int param_num_ = 0;

  const int minimum_log_size_ = 50;
  
  // Target Model parameter
  std::vector<double> target_model_param_; // [temp_target, temp_cloud, input, output, rssi, latency, error]

  int64_t EstimateInputSize(const Subgraph* subgraph);
  int64_t EstimateOutputSize(const Subgraph* subgraph);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_CLOUD_THERMAL_MODEL_H_