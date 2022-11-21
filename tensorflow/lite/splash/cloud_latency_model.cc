#include "tensorflow/lite/splash/cloud_latency_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/Cholesky"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;
using namespace Eigen;

TfLiteStatus CloudLatencyModel::Init() {
  model_param_ = vector<double>(3, 1.);
  window_size_ = 100;
  return kTfLiteOk;
}

int64_t CloudLatencyModel::Predict(Subgraph* subgraph) {
  int64_t comp_time = GetComputationTime(subgraph->GetKey().model_id);
  int64_t comm_time = PredictCommunicationTime(subgraph);
  return comp_time + comm_time;
}

int64_t CloudLatencyModel::GetComputationTime(int32_t model_id) {
  auto it = computation_time_table_.find(model_id);
  if (it != computation_time_table_.end()) {
    return it->second;
  } else {
    return 0; // Minimum value to be selected
  }
}

int64_t CloudLatencyModel::PredictCommunicationTime(Subgraph* subgraph) {
  int64_t comm_time = 2000; // minimum value
  vector<int64_t> regressor;
  if (log_size_ < 30) {
    return comm_time;
  }
  int64_t input_size = EstimateInputSize(subgraph);
  int64_t output_size = EstimateOutputSize(subgraph);
  regressor.push_back(input_size);
  regressor.push_back(output_size);
  regressor.push_back(1);
  if (regressor.size() != model_param_.size()) {
    return comm_time;
  }
  for (int i = 0; i < regressor.size(); i++) {
    comm_time += regressor[i] * model_param_[i]; 
  }
  return comm_time;
}

TfLiteStatus CloudLatencyModel::Profile(int32_t model_id, int64_t latency) {
  // Not implemented
  return kTfLiteOk;
}

TfLiteStatus CloudLatencyModel::Update(Job job, Subgraph* subgraph) {
  int64_t computation_time = job.latency - job.communication_time;
  auto it = computation_time_table_.find(job.model_id);
  if (it != computation_time_table_.end()) {
    int64_t prev_latency = computation_time_table_[job.model_id];
    computation_time_table_[job.model_id] =
        smoothing_factor_ * computation_time +
        (1 - smoothing_factor_) * prev_latency;
  } else {
    computation_time_table_[job.model_id] = computation_time;
  }
  UpdateCommunicationModel(subgraph, job.communication_time);
  return kTfLiteOk;
}

TfLiteStatus CloudLatencyModel::UpdateCommunicationModel(Subgraph* subgraph, int64_t communication_time) {
  int64_t input_size = EstimateInputSize(subgraph);
  int64_t output_size = EstimateOutputSize(subgraph);
  log_size_++;
  if (log_size_ <= window_size_) {
    X.conservativeResize(log_size_, 3);
    Y.conservativeResize(log_size_, 1);
  }
  int log_index = (log_size_ - 1) % window_size_;
  X.row(log_index) << input_size, output_size, 1.0;
  Y.row(log_index) << communication_time;

  if (log_size_ < 30) {
    LOGI("CloudLatencyModel::Update Not enough data : %d", log_size_);
    return kTfLiteOk;
  }
  Eigen::Matrix<double, 1, 3> theta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
  for (auto i = 0; i < model_param_.size(); i++) {
    model_param_[i] = theta(0, i); 
  }
  return kTfLiteOk;
}

int64_t CloudLatencyModel::EstimateInputSize(const Subgraph* subgraph) {
  // TODO: Add input tensors without weights.
  const std::vector<int>& input_tensors = subgraph->inputs();
  int64_t subgraph_input_size = 0;
  for (int tensor_idx : input_tensors) {
    subgraph_input_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_input_size;
}

int64_t CloudLatencyModel::EstimateOutputSize(const Subgraph* subgraph) {
  // TODO: Add output tensors without weights.
  const std::vector<int>& output_tensors = subgraph->outputs();
  int64_t subgraph_output_size = 0;
  for (int tensor_idx : output_tensors) {
    subgraph_output_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_output_size;
}

} // namespace impl
} // namespace tflite