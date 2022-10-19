#ifndef TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/resource_monitor.h"
#include "tensorflow/lite/splash/thermal_model.h"

namespace tflite {
namespace impl {

struct ThermalLog {
  explicit ThermalLog(Job job) {
    model_id = job.model_id;
    subgraph_idx = job.subgraph_idx;
    worker_id = job.worker_id;
    
    latency = job.latency;
    before_temp = job.before_temp;
    after_temp = job.after_temp;
    frequency = job.frequency;
    flops = job.flops;
    membytes = job.membytes;
  }

  int model_id;
  int subgraph_idx = -1; // For subgraph partitioning support in the future
  int worker_id = -1;

  int64_t latency = 0;
  std::vector<thermal_t> before_temp;
  std::vector<thermal_t> after_temp;
  std::vector<freq_t> frequency;

  // TODO : Remove these when using latency
  int64_t flops;
  int64_t membytes;
};

class ProcessorThermalModel : public IThermalModel {
 public:
  ProcessorThermalModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : IThermalModel(wid, resource_monitor) {}

  TfLiteStatus Init(int32_t worker_size, int32_t window_size) override;

  std::vector<thermal_t> Predict(const Subgraph* subgraph) override;

  TfLiteStatus Update(Job job) override;
 
 private:
  // Linear regressor
  std::vector<thermal_t> temp_regressor_; // Get from resource monitor
  std::vector<freq_t> freq_regressor_;
  std::int64_t flops_regressor_;
  std::int64_t membytes_regressor_;

  // Log buffer
  std::deque<ThermalLog> log_;
  int32_t window_size_;
  
  // Model parameter
  std::vector<std::vector<double>> model_param_; // worker_size * [temp_c, temp_g, temp_d, temp_n, freq_c, freq_g, latency, error]

  void PrintParameters();

  // TODO: Remove these methods
  int64_t EstimateFLOPS(const Subgraph* subgraph,
                        const Subgraph* primary_subgraph);
  int64_t EstimateInputOutputSize(const Subgraph* subgraph);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_