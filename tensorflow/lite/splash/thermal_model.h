#ifndef TENSORFLOW_LITE_THERMAL_MODEL_H_
#define TENSORFLOW_LITE_THERMAL_MODEL_H_

#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/splash/resource_monitor.h"

namespace tflite {
namespace impl {

class Subgraph;

class IThermalModel {
 public:
  IThermalModel(worker_id_t wid, 
                         ResourceMonitor& resource_monitor)
                         : wid_(wid), resource_monitor_(resource_monitor) {}

  // init model parameters with default values
  virtual TfLiteStatus Init(int32_t worker_size) = 0;

  // Get an estimation value of future temperature 
  // after executing inference of the input model
  virtual std::vector<thermal_t> Predict(const Subgraph* subgraph) = 0;

  // Update model parameters with the prediction error
  virtual TfLiteStatus Update(std::vector<thermal_t> error) = 0;

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

  std::vector<thermal_t> Multiply(const std::vector<thermal_t> &a, const int64_t &b)
  {
      const int n = a.size();     // a rows

      std::vector<thermal_t> c(n, 0);
      for (auto i = 0; i < n; ++i) {
        c[i] += a[i] * b;
      }
      return c;
  }

  // std::vector<thermal_t> Plus(const std::vector<thermal_t>& a, const std::vector<thermal_t>& b) {
  //   std::vector<thermal_t> result;
  //   result.reserve(a.size());

  //   std::transform(a.begin(), a.end(), b.begin(), 
  //                   std::back_inserter(result), std::plus<thermal_t>());
  //   return result;
  // }

  template <typename T>
  std::vector<T> Plus(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(), 
                    std::back_inserter(result), std::plus<T>());
    return result;
  }

 protected:
  worker_id_t wid_;
  ResourceMonitor& resource_monitor_;

  double gain_ = 0.0001;
};

} // namespace impl
} // namespace tflite

#endif // TENSORFLOW_LITE_THERMAL_MODEL_H_