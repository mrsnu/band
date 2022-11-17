#include "tensorflow/lite/splash/cloud_latency_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus CloudLatencyModel::Init() {
  return kTfLiteOk;
}

int64_t CloudLatencyModel::Predict(int32_t model_id) {
  int64_t comp_time = GetComputationTime(model_id);
  int64_t comm_time = PredictCommunicationTime(model_id);
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

int64_t CloudLatencyModel::PredictCommunicationTime(int32_t model_id) {
  auto it = computation_time_table_.find(model_id);
  if (it != computation_time_table_.end()) {
    return it->second;
  } else {
    return 0; // Minimum value to be selected
  }
}

int64_t CloudLatencyModel::PredictThrottled(int32_t model_id) {
  return Predict(model_id);
}

TfLiteStatus CloudLatencyModel::Profile(int32_t model_id, int64_t latency) {
  // Not implemented
  return kTfLiteOk;
}

TfLiteStatus CloudLatencyModel::Update(Job job) {
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