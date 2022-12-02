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
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;
using namespace Eigen;

TfLiteStatus ProcessorThermalModel::Init(ResourceConfig& config) {
  int temp_size = GetResourceMonitor().GetAllTemperature().size();
  int freq_size = GetResourceMonitor().GetAllFrequency().size();
  param_num_ = 1 + temp_size + freq_size + 2; 
  target_model_param_ = vector<double>(param_num_, 1.);
  window_size_ = config.model_update_window_size;
  model_path_ = config.thermal_model_param_path;
  LoadModelParameter(model_path_);
  return kTfLiteOk;
}

void ProcessorThermalModel::LoadModelParameter(string thermal_model_path) {
  LOGI("[ProcessorThermalModel] LoadModelParameter init");
  Json::Value model_param = LoadJsonObjectFromFile(thermal_model_path); 
  LOGI("[ProcessorThermalModel] load json done");
  for (auto worker_id_it = model_param.begin(); worker_id_it != model_param.end(); ++worker_id_it) {
    LOGI("[ProcessorThermalModel] here");
    int worker_id = std::atoi(worker_id_it.key().asString().c_str());
    LOGI("[ProcessorThermalModel] load worker %d", worker_id);
    if (worker_id != wid_) {
      continue;
    }

    const Json::Value param = *worker_id_it;
    for (auto it = param.begin(); it != param.end(); it++) {
      LOGI("[ProcessorThermalModel][%d] model_param : %f", it - param.begin(), (*it).asDouble());
      target_model_param_[it - param.begin()] = (*it).asDouble();
      is_thermal_model_prepared = true;
    }
  }
}


thermal_t ProcessorThermalModel::Predict(Subgraph* subgraph, 
                                         const int64_t latency, 
                                         std::vector<thermal_t> current_temp) {
  return PredictTarget(subgraph, latency, current_temp); 
}

thermal_t ProcessorThermalModel::PredictTarget(Subgraph* subgraph, 
                                         const int64_t latency, 
                                         std::vector<thermal_t> current_temp) {
  vector<double> regressor;
  thermal_t target_temp = GetResourceMonitor().GetTargetTemperature(wid_);
  if (!is_thermal_model_prepared) {
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

TfLiteStatus ProcessorThermalModel::Update(Job job, Subgraph* subgraph) {
  if (minimum_profiled_count_ < minimum_profiled_threshold_) {
    minimum_profiled_count_++;
    return kTfLiteOk;
  }
  log_size_++;
  if (log_size_ <= window_size_) {
    targetX.conservativeResize(log_size_, param_num_);
    targetY.conservativeResize(log_size_, 1);
  }
  int log_index = (log_size_ - 1) % window_size_;
  targetX.row(log_index) << job.before_target_temp[wid_], job.before_temp[0], job.before_temp[1], job.before_temp[2], job.before_temp[3], job.before_temp[4], job.frequency[0], job.frequency[1], job.latency, 1.0;
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
  Eigen::MatrixXd targetTheta; 
  targetTheta.conservativeResize(param_num_, 1);
  targetTheta = GetNormalEquation(targetX, targetY);
  for (auto i = 0; i < target_model_param_.size(); i++) {
    target_model_param_[i] = targetTheta(i, 0); 
  }
  is_thermal_model_prepared = true;
  return kTfLiteOk;
}

TfLiteStatus ProcessorThermalModel::Close() {
  if (!is_thermal_model_prepared) {
    return kTfLiteOk;
  }
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
