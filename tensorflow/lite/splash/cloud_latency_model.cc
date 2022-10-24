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
  // Get network bandwidth measurement 
  return 40000;
}

int64_t CloudLatencyModel::PredictThrottled(int32_t model_id) {
  // Get network bandwidth measurement 
  return 40000;
}

TfLiteStatus CloudLatencyModel::Update(int32_t model_id, int64_t latency) {
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