#include "tensorflow/lite/splash/processor_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;
using namespace Eigen;

TfLiteStatus ProcessorThermalModel::Init(int32_t window_size) {
  model_param_ = vector<double>(9, 1.);
  target_model_param_ = vector<double>(10, 1.);
  window_size_ = window_size;
  return kTfLiteOk;
}

thermal_t ProcessorThermalModel::Predict(const Subgraph* subgraph, 
                                         const int64_t latency, 
                                         std::vector<thermal_t> current_temp) {
  vector<int32_t> regressor;
  if (log_size_ < 50) {
    // Just return current temp
    return current_temp[wid_];
  }

  // Get temperature from resource monitor
  regressor.insert(regressor.end(), current_temp.begin(), current_temp.end());

  // Get frequency 
  vector<freq_t> freq = GetResourceMonitor().GetAllFrequency();
  regressor.insert(regressor.end(), freq.begin(), freq.end());

  regressor.push_back(latency);
  regressor.push_back(1);

  thermal_t future_temperature = 0;

  if (regressor.size() != model_param_.size()) {
    LOGI("[ProcessorThermalModel] Error!!: regressor.size()[%d] != model_param_.size()[%d]", regressor.size(), model_param_.size());
    return future_temperature;
  }

  for (int i = 0; i < regressor.size(); i++) {
    future_temperature += regressor[i] * model_param_[i];
  }
  return future_temperature; 
}

thermal_t ProcessorThermalModel::PredictTarget(const Subgraph* subgraph, 
                                         const int64_t latency, 
                                         std::vector<thermal_t> current_temp) {
  vector<int32_t> regressor;
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_);
  if (log_size_ < 50) {
    // Just return current temp
    return target_temp;
  }

  regressor.push_back(target_temp);
  regressor.insert(regressor.end(), current_temp.begin(), current_temp.end());
  vector<freq_t> freq = GetResourceMonitor().GetAllFrequency();
  regressor.insert(regressor.end(), freq.begin(), freq.end());
  regressor.push_back(latency);
  regressor.push_back(1);

  thermal_t target_future_temperature = 0;

  if (regressor.size() != target_model_param_.size()) {
    LOGI("[Error!!: regressor.size()[%d] != target_model_param_.size()[%d]", regressor.size(), target_model_param_.size());
    return target_future_temperature;
  }

  for (int i = 0; i < regressor.size(); i++) {
    target_future_temperature += regressor[i] * target_model_param_[i]; 
  }
  return target_future_temperature; 
}

void ProcessorThermalModel::PrintParameters() {
  LOGI("================Temp Param(S)================");
  std::stringstream ss;
  for (auto i = 0; i < model_param_.size(); ++i) {
    ss << model_param_[i] << '\t';
  }
  LOGI("%s", ss.str().c_str());
  LOGI("================Temp Param(E)================");
}

TfLiteStatus ProcessorThermalModel::Update(Job job) {
  log_size_++;
  if (log_size_ <= window_size_) {
    X.conservativeResize(log_size_, 9);
    targetX.conservativeResize(log_size_, 10);
    Y.conservativeResize(log_size_, 1);
    targetY.conservativeResize(log_size_, 1);
  }
  int log_index = (log_size_ - 1) % window_size_;
  X.row(log_index) << job.before_temp[0], job.before_temp[1], job.before_temp[2], job.before_temp[3], job.before_temp[4], job.frequency[0], job.frequency[1], job.latency, 1.0;
  targetX.row(log_index) << job.before_target_temp[wid_], job.before_temp[0], job.before_temp[1], job.before_temp[2], job.before_temp[3], job.before_temp[4], job.frequency[0], job.frequency[1], job.latency, 1.0;
  Y.row(log_index) << job.after_temp[wid_];
  targetY.row(log_index) << job.after_target_temp[wid_];

  if (log_size_ < 50) {
    LOGI("ProcessorThermalModel::Update Not enough data : %d", log_size_);
    return kTfLiteOk;
  }

  // Update parameters via normal equation with log table
  Eigen::Matrix<double, 1, 9> theta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
  Eigen::Matrix<double, 1, 10> targetTheta = (targetX.transpose() * targetX).ldlt().solve(targetX.transpose() * targetY);
  for (auto i = 0; i < model_param_.size(); i++) {
    model_param_[i] = theta(0, i); 
  }
  for (auto i = 0; i < target_model_param_.size(); i++) {
    target_model_param_[i] = targetTheta(0, i); 
  }
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite