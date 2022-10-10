#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

TfLiteStatus CloudThermalModel::Init() {
    return kTfLiteOk;
}

std::unordered_map<worker_id_t, ThermalInfo> CloudThermalModel::Predict(SubgraphKey key) {
  std::unordered_map<worker_id_t, ThermalInfo> future_temperature;
  return future_temperature;
}

TfLiteStatus CloudThermalModel::Update(std::vector<ThermalInfo> error) {
  return kTfLiteOk;
}


} // namespace impl
} // namespace tflite