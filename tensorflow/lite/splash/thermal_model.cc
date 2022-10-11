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
    models_.push_back(BuildModel(i));
  }
  for (auto& model : models_) {
    auto status = model->Init(models_.size());
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

IThermalModel * ThermalModel::BuildModel(worker_id_t wid) {
  switch (wid) {
    case kTfLiteCPU:
      return new ProcessorThermalModel(wid, resource_monitor_);
    case kTfLiteGPU:
      return new ProcessorThermalModel(wid, resource_monitor_);
    case kTfLiteDSP:
      return new ProcessorThermalModel(wid, resource_monitor_);
    case kTfLiteNPU:
      return new ProcessorThermalModel(wid, resource_monitor_);
    case kTfLiteCLOUD:
      return new CloudThermalModel(wid, resource_monitor_);
    default:
      return new ProcessorThermalModel(wid, resource_monitor_);
  }
}

std::vector<worker_id_t> ThermalModel::GetPossibleWorkers(Subgraph* subgraph) {
  std::vector<worker_id_t> possible_workers;
  for (auto& model : models_) {
    auto temperature = model->Predict(subgraph);
    bool throttled = false;
    for (int i = 0; i < temperature.size(); i++) {
      thermal_t temp = temperature[i];
      // Checks if throttled
      auto threshold = resource_monitor_.GetThrottlingThreshold(i);
      if (temp > threshold) {
        throttled = true;
        break;
      }
    }
    if (!throttled) {
      possible_workers.push_back(model->GetWorkerId());
    }
  }
  return possible_workers;
}

std::vector<thermal_t> ThermalModel::GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph) {
  return models_[wid]->Predict(subgraph);
}

TfLiteStatus ThermalModel::Update(std::vector<thermal_t> error, worker_id_t wid) {
  return models_[wid]->Update(error);
}


} // namespace impl
} // namespace tflite