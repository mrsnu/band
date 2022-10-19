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

class ProcessorThermalModel : public IThermalModel {
 public:
  ProcessorThermalModel(worker_id_t wid, ResourceMonitor& resource_monitor)
  : IThermalModel(wid, resource_monitor) {}

  TfLiteStatus Init(int32_t worker_size) override;

  std::vector<thermal_t> Predict(const Subgraph* subgraph) override;

  TfLiteStatus Update(std::vector<thermal_t> error) override;
 
 private:
  // Linear regressor
  std::vector<thermal_t> temp_regressor_; // Get from resource monitor
  std::vector<freq_t> freq_regressor_;
  std::int64_t flops_regressor_;
  std::int64_t membytes_regressor_;
  
  // Model parameter
  std::vector<std::vector<double>> temp_param_;
  std::vector<std::vector<double>> freq_param_;
  std::vector<double> flops_param_;
  std::vector<double> membytes_param_;
  std::vector<double> error_param_;


  void UpdateParameters(std::vector<std::vector<double>>& params, 
                      std::vector<thermal_t> error, 
                      std::vector<thermal_t> regressor);

  void UpdateParameters(std::vector<double>& params, 
                      std::vector<thermal_t> error,
                      int64_t regressor);

  void PrintParameters();

  std::vector<thermal_t> EstimateFutureTemperature(const std::vector<thermal_t> temp,
                                                   const std::vector<freq_t> freq,
                                                   const int64_t flops,
                                                   const int64_t membytes);
  int64_t EstimateFLOPS(const Subgraph* subgraph,
                        const Subgraph* primary_subgraph);
  int64_t EstimateInputOutputSize(const Subgraph* subgraph);
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_PROCESSOR_THERMAL_MODEL_H_