#ifndef TENSORFLOW_LITE_CLOUD_LATENCY_MODEL_H_
#define TENSORFLOW_LITE_CLOUD_LATENCY_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/latency_model.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"

namespace tflite {
namespace impl {

class CloudLatencyModel : public ILatencyModel {
 public:
  CloudLatencyModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : ILatencyModel(wid, resource_monitor) {}

  TfLiteStatus Init(ResourceConfig& config) override;

  int64_t Predict(Subgraph* subgraph) override;

  TfLiteStatus Update(Job job, Subgraph* subgraph) override;
  
  TfLiteStatus Profile(int32_t model_id, int64_t latency) override;

 private:
  std::unordered_map<std::string, int64_t> computation_time_table_; // {model_name, latency}
  int64_t communication_time_;

  // Log buffer
  Eigen::MatrixXd X;
  Eigen::VectorXd Y;
  uint32_t log_size_ = 0;
  int32_t window_size_;
  
  // Model parameter
  std::vector<double> model_param_; // [input, output, error]
  void LoadModelParameter(std::string latency_model_path);
  int64_t EstimateInputSize(const Subgraph* subgraph);
  int64_t EstimateOutputSize(const Subgraph* subgraph);
  int64_t GetComputationTime(std::string model_name);
  int64_t PredictCommunicationTime(Subgraph* subgraph);
  TfLiteStatus UpdateCommunicationModel(Subgraph* subgraph, int64_t communication_time);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_CLOUD_LATENCY_MODEL_H_