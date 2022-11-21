#ifndef TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_

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

class ProcessorThermalModel : public IThermalModel {
 public:
  ProcessorThermalModel(worker_id_t wid, ResourceMonitor& resource_monitor)
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
  // Log buffer
  Eigen::MatrixXd X;
  Eigen::MatrixXd targetX;
  Eigen::VectorXd Y;
  Eigen::VectorXd targetY;
  uint32_t log_size_ = 0;
  int32_t window_size_;
  
  // Model parameter
  std::vector<double> model_param_; // [temp_c, temp_g, temp_d, temp_n, freq_c, freq_g, latency, error]

  // Target Model parameter
  std::vector<double> target_model_param_; // Same structure with model_param_

  void PrintParameters();
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_