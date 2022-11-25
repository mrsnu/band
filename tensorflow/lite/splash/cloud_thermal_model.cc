#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

#define TARGET_PARAM_NUM 7

namespace tflite {
namespace impl {

using namespace std;
using namespace Eigen;

TfLiteStatus CloudThermalModel::Init(ResourceConfig& config) {
  window_size_ = config.model_update_window_size;
  target_model_param_ = vector<double>(TARGET_PARAM_NUM, 1.);
  model_path_ = config.thermal_model_param_path;
  LoadModelParameter(model_path_);
  return kTfLiteOk;
}

void CloudThermalModel::LoadModelParameter(string thermal_model_path) {
  Json::Value model_param = LoadJsonObjectFromFile(thermal_model_path); 
  for (auto worker_id_it = model_param.begin(); worker_id_it != model_param.end(); ++worker_id_it) {
    int worker_id = std::atoi(worker_id_it.key().asString().c_str());
    if (worker_id != wid_) {
      continue;
    }

    const Json::Value param = *worker_id_it;
    for (auto it = param.begin(); it != param.end(); it++) {
      LOGI("[CloudThermalModel][%d] model_param : %f", it - param.begin(), (*it).asDouble());
      target_model_param_[it - param.begin()] = (*it).asDouble();
    }
  }
}

thermal_t CloudThermalModel::Predict(const Subgraph* subgraph, 
                                     const int64_t latency, 
                                     std::vector<thermal_t> current_temp) {
  return PredictTarget(subgraph, latency, current_temp);
}

thermal_t CloudThermalModel::PredictTarget(const Subgraph* subgraph, 
                                     const int64_t latency, 
                                     std::vector<thermal_t> current_temp) {
  vector<double> regressor;
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_);
  if (log_size_ < minimum_update_log_size_) {
    // Just return current temp
    return target_temp;
  }

  regressor.push_back(target_temp);
  regressor.push_back(current_temp[wid_]);
  regressor.push_back(EstimateInputSize(subgraph));
  regressor.push_back(EstimateOutputSize(subgraph));
  regressor.push_back(-49); // RSSI value
  regressor.push_back(latency);
  regressor.push_back(1);

  double target_future_temperature = 0.;

  if (regressor.size() != target_model_param_.size()) {
    LOGI("[Error!!: regressor.size()[%d] != target_model_param_.size()[%d]", regressor.size(), target_model_param_.size());
    return target_future_temperature;
  }

  for (int i = 0; i < regressor.size(); i++) {
    target_future_temperature += regressor[i] * target_model_param_[i]; 
  }
  return (thermal_t)target_future_temperature; 
}

int64_t CloudThermalModel::EstimateInputSize(const Subgraph* subgraph) {
  const std::vector<int>& input_tensors = subgraph->inputs();
  int64_t subgraph_input_size = 0;
  for (int tensor_idx : input_tensors) {
    subgraph_input_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_input_size;
}

int64_t CloudThermalModel::EstimateOutputSize(const Subgraph* subgraph) {
  const std::vector<int>& output_tensors = subgraph->outputs();
  int64_t subgraph_output_size = 0;
  for (int tensor_idx : output_tensors) {
    subgraph_output_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_output_size;
}

TfLiteStatus CloudThermalModel::Update(Job job, const Subgraph* subgraph) {
  log_size_++;
  if (log_size_ <= window_size_) {
    targetX.conservativeResize(log_size_, TARGET_PARAM_NUM);
    targetY.conservativeResize(log_size_, 1);
  }
  int log_index = (log_size_ - 1) % window_size_;
  targetX.row(log_index) << job.before_target_temp[wid_], job.before_temp[wid_], EstimateInputSize(subgraph), EstimateOutputSize(subgraph), -49, job.latency, 1.0; // RSSI value
  targetY.row(log_index) << job.after_target_temp[wid_];

  if (log_size_ < minimum_log_size_) {
    LOGI("CloudThermalModel::Update Not enough data : %d", log_size_);
    return kTfLiteOk;
  }

  // Update parameters via normal equation with log table
  Eigen::Matrix<double, 1, TARGET_PARAM_NUM> targetTheta = (targetX.transpose() * targetX).ldlt().solve(targetX.transpose() * targetY);
  for (auto i = 0; i < target_model_param_.size(); i++) {
    target_model_param_[i] = targetTheta(0, i); 
  }
  return kTfLiteOk;
}

TfLiteStatus CloudThermalModel::Close() {
  Json::Value root= LoadJsonObjectFromFile(model_path_); 
  Json::Value param;
  for (int i = 0; i < target_model_param_.size(); i++) {
    param.append(target_model_param_[i]); 
  } 
  root[std::to_string(wid_)] = param;
  WriteJsonObjectToFile(root, model_path_);
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite
