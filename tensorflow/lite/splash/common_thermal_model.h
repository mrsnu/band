#ifndef TENSORFLOW_LITE_COMMON_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_COMMON_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/splash/resource_monitor.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

class CommonThermalModel {
 public:
  CommonThermalModel(worker_id_t wid, ResourceMonitor& resource_monitor): wid_(wid), resource_monitor_(resource_monitor) {
    LOGI("make instance of common thermal model : %d", wid);
  }

  TfLiteStatus Init(int32_t worker_size);

  std::vector<thermal_t> Predict(const Subgraph* subgraph);

  TfLiteStatus Update(std::vector<thermal_t> error);

  worker_id_t GetWorkerId() {
    return wid_;
  }

  ResourceMonitor& GetResourceMonitor() {
    return resource_monitor_;
  }

  std::vector<thermal_t> Multiply(const std::vector<std::vector<double>> &a, const std::vector<thermal_t> &b)
  {
      const int n = a.size();     // a rows
      const int m = a[0].size();  // a cols

      std::vector<thermal_t> c(n, 0);
      for (auto k = 0; k < m; ++k) {
        for (auto i = 0; i < n; ++i) {
          c[i] += (thermal_t) (a[i][k] * b[k]);
        }
      }
      return c;
  }

  std::vector<thermal_t> Multiply(const std::vector<double> &a, const int64_t &b)
  {
      const int n = a.size();     // a rows

      std::vector<thermal_t> c(n, 0);
      for (auto i = 0; i < n; ++i) {
        c[i] += a[i] * b;
      }
      return c;
  }

  std::vector<thermal_t> Plus(const std::vector<thermal_t>& a, const std::vector<thermal_t>& b) {
    std::vector<thermal_t> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                    std::back_inserter(result), std::plus<thermal_t>());
    return result;
  }
 
 private:
  worker_id_t wid_;
  ResourceMonitor& resource_monitor_;

  // Linear regressor
  std::vector<int32_t> temperature; // Get from resource monitor
  std::vector<int32_t> frequency;
  std::int64_t flops;
  std::int64_t membytes;
  
  // Model parameter
  std::vector<std::vector<double>> temp_param_;
  std::vector<std::vector<double>> freq_param_;
  std::vector<double> flops_param_;
  std::vector<double> membytes_param_;
  std::vector<double> error_param_;

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

#endif // TENSORFLOW_LITE_COMMON_THERMAL_MODEL_H_