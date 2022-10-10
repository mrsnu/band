#include "tensorflow/lite/splash/thermal_model.h"
#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include "tensorflow/lite/splash/processor_thermal_model.h"

#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

TfLiteStatus ThermalModel::Init() {
  for (int i = 0; i < kTfLiteNumDevices; i++) {
    if (i == kTfLiteOffloading) {
      models_["Cloud"] = new CloudThermalModel();
    } else if (i == kTfLiteCPU) {
      models_["CPU"] = new ProcessorThermalModel();
    } else if (i == kTfLiteGPU) {
      models_["GPU"] = new ProcessorThermalModel();
    } else if (i == kTfLiteDSP) {
      models_["DSP"] = new ProcessorThermalModel();
    } else if (i == kTfLiteNPU) {
      models_["NPU"] = new ProcessorThermalModel();
    }
  }
  for (auto& model : models_) {
    auto status = model.second->Init();
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

std::vector<worker_id_t> ThermalModel::GetPossibleWorkers(SubgraphKey key) {
  std::vector<worker_id_t> possible_workers;
  for (auto& model : models_) {
    auto temperature = model.second->Predict(key);
    bool throttled = false;
    for (auto& temp : temperature) {
      // Checks if throttled
      auto threshold = resource_monitor_.GetTemperature(temp.first);
      if (temp.second.temperature > threshold) {
        throttled = true;
        break;
      }
    }
    if (!throttled) {
      possible_workers.push_back(model.first);
    }
  }
  return possible_workers;
}

TfLiteStatus ThermalModel::Update(std::vector<ThermalInfo> error, worker_id_t wid) {
  return models_[wid]->Update(error);
}


} // namespace impl
} // namespace tflite