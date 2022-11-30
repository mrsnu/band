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

  TfLiteStatus Init(ResourceConfig& config) override;

  thermal_t Predict(const Subgraph* subgraph, 
                    const int64_t latency, 
                    std::vector<thermal_t> current_temp) override;

  thermal_t PredictTarget(const Subgraph* subgraph, 
                    const int64_t latency, 
                    std::vector<thermal_t> current_temp) override;

  TfLiteStatus Update(Job job, const Subgraph* subgraph) override;

  TfLiteStatus Close() override;
 
 private:
  // Log buffer
  Eigen::MatrixXd X;
  Eigen::MatrixXd targetX;
  Eigen::VectorXd Y;
  Eigen::VectorXd targetY;
  uint32_t log_size_ = 0;
  int window_size_;
  int param_num_ = 0;

  bool is_thermal_model_prepared = false; 
  const int minimum_update_log_size_ = 50;
  int minimum_profiled_count_ = 0; 
  int minimum_profiled_threshold_ = 5;
  std::string model_path_;
  
  // Model parameter
  std::vector<double> model_param_; // [temp_c, temp_g, temp_d, temp_n, temp_cloud, freq_c, freq_g, latency, error]

  // Target Model parameter
  std::vector<double> target_model_param_; // temp_target + [model_param_]

  void LoadModelParameter(string thermal_model_path);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_