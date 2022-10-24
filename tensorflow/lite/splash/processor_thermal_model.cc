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

TfLiteStatus ProcessorThermalModel::Init(int32_t worker_size, int32_t window_size) {
  model_param_.assign(worker_size, vector<double>(9, 1.));
  window_size_ = window_size;
  return kTfLiteOk;
}

vector<thermal_t> ProcessorThermalModel::Predict(const Subgraph* subgraph, 
                                                 const int64_t latency, 
                                                 std::vector<thermal_t> current_temp) {
  vector<int32_t> regressor;
  if (log_.size() < 50) {
    // Just return current temp
    return current_temp;
  }

  // Get temperature from resource monitor
  regressor.insert(regressor.end(), current_temp.begin(), current_temp.end());

  // Get frequency 
  vector<freq_t> freq = GetResourceMonitor().GetAllFrequency();
  regressor.insert(regressor.end(), freq.begin(), freq.end());

  regressor.push_back(latency);

  // Push error term
  regressor.push_back(1);

  vector<thermal_t> future_temperature(model_param_.size(), 0);

  if (regressor.size() != model_param_[0].size()) {
    LOGI("[ProcessorThermalModel] Error!!: regressor.size()[%d] != model_param_.size()[%d]", regressor.size(), model_param_[0].size());
    return future_temperature;
  }

  for (int wid = 0; wid < model_param_.size(); wid++) {
    thermal_t predicted = 0;
    for (int i = 0; i < regressor.size(); i++) {
      predicted += regressor[i] * model_param_[wid][i];
    }
    future_temperature[wid] = predicted;
  }
  return future_temperature; 
}

void ProcessorThermalModel::PrintParameters() {
  LOGI("================Temp Param(S)================");
  for (auto wid = 0; wid < model_param_.size(); ++wid) {
    std::stringstream ss;
    for (auto i = 0; i < model_param_[wid].size(); ++i) {
      ss << model_param_[wid][i] << '\t';
    }
    LOGI("%s", ss.str().c_str());
  }
  LOGI("================Temp Param(E)================");
}

TfLiteStatus ProcessorThermalModel::Update(Job job) {
  // Add job to log table
  ThermalLog log(job);
  if (log_.size() > window_size_) {
    log_.pop_front();
  }
  log_.push_back(log);

  if (log_.size() < 50) {
    LOGI("ProcessorThermalModel::Update Not enough data : %d", log_.size());
    return kTfLiteOk;
  }

  // Update parameters via normal equation with log table
  for (auto target_worker = 0; target_worker < model_param_.size(); target_worker++) {
    Eigen::MatrixXd X;
    X.resize(log_.size(), 9);
    Eigen::VectorXd Y;
    Y.resize(log_.size(), 1);
    for (auto i = 0; i < log_.size(); i++) {
      // TODO : init matrix with vector more efficient way
      X(i, 0) = log_[i].before_temp[0];
      X(i, 1) = log_[i].before_temp[1];
      X(i, 2) = log_[i].before_temp[2];
      X(i, 3) = log_[i].before_temp[3];
      X(i, 4) = log_[i].before_temp[4];
      X(i, 5) = log_[i].frequency[0];
      X(i, 6) = log_[i].frequency[1];
      X(i, 7) = log_[i].latency;
      X(i, 8) = 1.0;
      Y(i, 0) = log_[i].after_temp[target_worker];
    }
    Eigen::Matrix<double, 1, 9> theta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
    for (auto i = 0; i < model_param_[target_worker].size(); i++) {
      model_param_[target_worker][i] = theta(0, i); 
    }
  }

  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite