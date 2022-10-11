#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus CloudThermalModel::Init(int32_t worker_size) {
  temp_param_.assign(worker_size, vector<double>(worker_size, 1.0));
  input_param_.assign(worker_size, 1.0);
  output_param_.assign(worker_size, 1.0);
  rssi_param_.assign(worker_size, 1.0);
  waiting_param_.assign(worker_size, 1.0);
  error_param_.assign(worker_size, 1.0);
  return kTfLiteOk;
}

vector<thermal_t> CloudThermalModel::Predict(const Subgraph* subgraph) {
  // Get temperature from resource monitor
  vector<thermal_t> temp = GetResourceMonitor().GetAllTemperature();

  // Get input size
  int64_t input_size = EstimateInputSize(subgraph);

  // Get ouput size
  int64_t output_size = EstimateOutputSize(subgraph);

  // Get rssi value from resource monitor
  int64_t rssi = -50;

  // Get expected latency from server from resource monitor
  int64_t waiting_time = GetResourceMonitor().GetThrottlingThreshold(GetWorkerId());

  return EstimateFutureTemperature(temp, input_size, output_size, rssi, waiting_time);
}

vector<thermal_t> CloudThermalModel::EstimateFutureTemperature(const vector<thermal_t> temp,
                                                                   const int64_t input_size,
                                                                   const int64_t output_size,
                                                                   const int64_t rssi,
                                                                   const int64_t waiting_time) {
  vector<thermal_t> future_temperature;
  // future_temperature = temp * temp_param_ + input_size * input_param_ + output_size * output_param_ + rssi * rssi_param_ + waiting_time * waiting_param_ + 1 * error_param_;
  return future_temperature;
}

int64_t CloudThermalModel::EstimateInputSize(const Subgraph* subgraph) {
  // TODO: Add input tensors without weights.
  const std::vector<int>& input_tensors = subgraph->inputs();
  int64_t subgraph_input_size = 0;
  for (int tensor_idx : input_tensors) {
    subgraph_input_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_input_size;
}

int64_t CloudThermalModel::EstimateOutputSize(const Subgraph* subgraph) {
  // TODO: Add output tensors without weights.
  const std::vector<int>& output_tensors = subgraph->outputs();
  int64_t subgraph_output_size = 0;
  for (int tensor_idx : output_tensors) {
    subgraph_output_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_output_size;
}

TfLiteStatus CloudThermalModel::Update(vector<thermal_t> error) {
  return kTfLiteOk;
}


} // namespace impl
} // namespace tflite