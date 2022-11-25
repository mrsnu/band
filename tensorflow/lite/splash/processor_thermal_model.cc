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

#define PARAM_NUM 9
#define TARGET_PARAM_NUM 10

namespace tflite {
namespace impl {

using namespace std;
using namespace Eigen;

TfLiteStatus ProcessorThermalModel::Init(ResourceConfig& config) {
  int temp_size = GetResourceMonitor().GetAllTemperature().size();
  int freq_size = GetResourceMonitor().GetAllFrequency().size();
  int param_num = temp_size + freq_size + 2; 
  if (param_num != PARAM_NUM) {
    LOGI("[Error] ProcessorThermalModel - param size error");
  }
  model_param_ = vector<double>(PARAM_NUM, 1.);
  target_model_param_ = vector<double>(TARGET_PARAM_NUM, 1.);
  window_size_ = config.model_update_window_size;
  model_path_ = config.thermal_model_param_path;
  LoadModelParameter(model_path_);
  return kTfLiteOk;
}

void ProcessorThermalModel::LoadModelParameter(string thermal_model_path) {
  Json::Value model_param = LoadJsonObjectFromFile(thermal_model_path); 
  for (auto worker_id_it = model_param.begin(); worker_id_it != model_param.end(); ++worker_id_it) {
    int worker_id = std::atoi(worker_id_it.key().asString().c_str());
    if (worker_id != wid_) {
      continue;
    }

    const Json::Value param = *worker_id_it;
    for (auto it = param.begin(); it != param.end(); it++) {
      LOGI("[ProcessorThermalModel][%d] model_param : %f", it - param.begin(), (*it).asDouble());
      target_model_param_[it - param.begin()] = (*it).asDouble();
    }
  }
}


thermal_t ProcessorThermalModel::Predict(const Subgraph* subgraph, 
                                         const int64_t latency, 
                                         std::vector<thermal_t> current_temp) {
  vector<double> regressor;
  if (log_size_ < minimum_log_size_) {
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

  double future_temperature = 0;

  if (regressor.size() != model_param_.size()) {
    LOGI("[ProcessorThermalModel] Error!!: regressor.size()[%d] != model_param_.size()[%d]", regressor.size(), model_param_.size());
    return future_temperature;
  }

  for (int i = 0; i < regressor.size(); i++) {
    future_temperature += regressor[i] * model_param_[i];
  }
  return (thermal_t) future_temperature; 
}

thermal_t ProcessorThermalModel::PredictTarget(const Subgraph* subgraph, 
                                         const int64_t latency, 
                                         std::vector<thermal_t> current_temp) {
  vector<double> regressor;
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_);
  if (log_size_ < minimum_log_size_) {
    // Just return current temp
    return target_temp;
  }

  regressor.push_back(target_temp);
  regressor.insert(regressor.end(), current_temp.begin(), current_temp.end());
  vector<freq_t> freq = GetResourceMonitor().GetAllFrequency();
  regressor.insert(regressor.end(), freq.begin(), freq.end());
  regressor.push_back(latency);
  regressor.push_back(1);

  double target_future_temperature = 0;

  if (regressor.size() != target_model_param_.size()) {
    LOGI("[Error!!: regressor.size()[%d] != target_model_param_.size()[%d]", regressor.size(), target_model_param_.size());
    return target_future_temperature;
  }

  for (int i = 0; i < regressor.size(); i++) {
    target_future_temperature += regressor[i] * target_model_param_[i]; 
  }
  return (thermal_t) target_future_temperature; 
}

TfLiteStatus ProcessorThermalModel::Update(Job job, const Subgraph* subgraph) {
  log_size_++;
  if (log_size_ <= window_size_) {
    X.conservativeResize(log_size_, PARAM_NUM);
    targetX.conservativeResize(log_size_, TARGET_PARAM_NUM);
    Y.conservativeResize(log_size_, 1);
    targetY.conservativeResize(log_size_, 1);
  }
  int log_index = (log_size_ - 1) % window_size_;
  X.row(log_index) << job.before_temp[0], job.before_temp[1], job.before_temp[2], job.before_temp[3], job.before_temp[4], job.frequency[0], job.frequency[1], job.latency, 1.0;
  targetX.row(log_index) << job.before_target_temp[wid_], job.before_temp[0], job.before_temp[1], job.before_temp[2], job.before_temp[3], job.before_temp[4], job.frequency[0], job.frequency[1], job.latency, 1.0;
  if (job.after_temp[wid_] < job.before_temp[wid_]) {
    Y.row(log_index) << job.before_temp[wid_]; 
  } else {
    Y.row(log_index) << job.after_temp[wid_];
  }
  if (job.after_target_temp[wid_] < job.before_target_temp[wid_]) {
    targetY.row(log_index) << job.before_target_temp[wid_];
  } else {
    targetY.row(log_index) << job.after_target_temp[wid_];
  }

  if (log_size_ < minimum_update_log_size_) {
    LOGI("ProcessorThermalModel::Update Not enough data : %d", log_size_);
    return kTfLiteOk;
  }

  // Update parameters via normal equation with log table
  Eigen::Matrix<double, 1, PARAM_NUM> theta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
  Eigen::Matrix<double, 1, TARGET_PARAM_NUM> targetTheta = (targetX.transpose() * targetX).ldlt().solve(targetX.transpose() * targetY);
  for (auto i = 0; i < model_param_.size(); i++) {
    model_param_[i] = theta(0, i); 
  }
  for (auto i = 0; i < target_model_param_.size(); i++) {
    target_model_param_[i] = targetTheta(0, i); 
  }
  return kTfLiteOk;
}

TfLiteStatus ProcessorThermalModel::Close() {
  Json::Value root;
  if (wid_ != 0) {
    root = LoadJsonObjectFromFile(model_path_); 
  }
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
