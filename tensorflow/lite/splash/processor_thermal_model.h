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
  }

  int model_id;
  int subgraph_idx = -1; // For subgraph partitioning support in the future
  int worker_id = -1;

  int64_t latency = 0;
  std::vector<thermal_t> before_temp;
  std::vector<thermal_t> after_temp;
  std::vector<freq_t> frequency;
};

class ProcessorThermalModel : public IThermalModel {
 public:
  ProcessorThermalModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : IThermalModel(wid, resource_monitor) {}

  TfLiteStatus Init(int32_t window_size) override;

  thermal_t Predict(const Subgraph* subgraph, 
                    const int64_t latency, 
                    std::vector<thermal_t> current_temp) override;

  TfLiteStatus Update(Job job) override;
 
 private:
  // Log buffer
  std::deque<ThermalLog> log_;
  int32_t window_size_;
  
  // Model parameter
  std::vector<double> model_param_; // [temp_c, temp_g, temp_d, temp_n, freq_c, freq_g, latency, error]

  void PrintParameters();
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_