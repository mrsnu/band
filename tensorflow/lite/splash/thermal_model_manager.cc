#include "tensorflow/lite/splash/thermal_model_manager.h"

#include "tensorflow/lite/splash/thermal_model.h"
#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include "tensorflow/lite/splash/processor_thermal_model.h"
#include "tensorflow/lite/profiling/time.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

std::unique_ptr<IThermalModel> ThermalModelManager::BuildModel(worker_id_t wid) {
  switch (wid) {
    case kTfLiteCPU:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteGPU:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteDSP:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteNPU:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteCLOUD:
      return std::make_unique<CloudThermalModel>(wid, resource_monitor_);
    default:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
  }
}

TfLiteStatus ThermalModelManager::Init() {
  LOGI("ThermalModelManager:: init");
  // cpu_model_ = BuildModel(0);

  // for (int wid = 0; wid < kTfLiteNumDevices; wid++) {
  //   std::unique_ptr<IThermalModel> model = BuildModel(wid);
  //   models_.emplace_back(std::move(model));
  // }
  // for (auto& model : models_) {
  //   auto status = model->Init(models_.size());
  //   if (status == kTfLiteError) {
  //     return kTfLiteError;
  //   }
  // }
  LOGI("ThermalModelManager:: finish");
  return kTfLiteOk;
}

std::vector<worker_id_t> ThermalModelManager::GetPossibleWorkers(Subgraph* subgraph) {
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

std::vector<thermal_t> ThermalModelManager::GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph) {
  LOGI("GetPredictedTemperature starts : %d", wid);
  return models_[wid]->Predict(subgraph);
}

TfLiteStatus ThermalModelManager::Update(std::vector<thermal_t> error, worker_id_t wid) {
  return models_[wid]->Update(error);
}


} // namespace impl
} // namespace tflite