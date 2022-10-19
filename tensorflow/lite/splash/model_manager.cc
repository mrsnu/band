#include "tensorflow/lite/splash/model_manager.h"

#include "tensorflow/lite/splash/thermal_model.h"
#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include "tensorflow/lite/splash/processor_thermal_model.h"
#include "tensorflow/lite/splash/latency_model.h"
#include "tensorflow/lite/splash/cloud_latency_model.h"
#include "tensorflow/lite/splash/processor_latency_model.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {


TfLiteStatus ModelManager::Init(ResourceConfig& config) {
  LOGI("ModelManager:: init");
  // Build ThermalModel
  for (int wid = 0; wid < kTfLiteNumDevices; wid++) {
    std::unique_ptr<IThermalModel> model = BuildThermalModel(wid);
    thermal_models_.emplace_back(std::move(model));
  }
  for (auto& thermal_model : thermal_models_) {
    auto status = thermal_model->Init(thermal_models_.size(), config.model_update_window_size);
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }

  // Build LatencyModel
  for (int wid = 0; wid < kTfLiteNumDevices; wid++) {
    std::unique_ptr<ILatencyModel> model = BuildLatencyModel(wid);
    latency_models_.emplace_back(std::move(model));
  }
  for (auto& latency_model : latency_models_) {
    auto status = latency_model->Init();
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }
  LOGI("ModelManager:: finish");
  return kTfLiteOk;
}

std::unique_ptr<IThermalModel> ModelManager::BuildThermalModel(worker_id_t wid) {
  if (wid == kTfLiteCLOUD) {
    return std::make_unique<CloudThermalModel>(wid, resource_monitor_);
  }
  return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
}

std::unique_ptr<ILatencyModel> ModelManager::BuildLatencyModel(worker_id_t wid) {
  if (wid == kTfLiteCLOUD) {
    return std::make_unique<CloudLatencyModel>(wid);
  }
  return std::make_unique<ProcessorLatencyModel>(wid);
}

std::vector<worker_id_t> ModelManager::GetPossibleWorkers(Subgraph* subgraph) {
  std::vector<worker_id_t> possible_workers;
  for (auto& thermal_model : thermal_models_) {
    auto temperature = thermal_model->Predict(subgraph);
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
      possible_workers.push_back(thermal_model->GetWorkerId());
    }
  }
  return possible_workers;
}

std::vector<thermal_t> ModelManager::GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph) {
  LOGI("GetPredictedTemperature starts : %d", wid);
  return thermal_models_[wid]->Predict(subgraph);
}

int64_t ModelManager::GetPredictedLatency(worker_id_t wid, int32_t model_id) {
  LOGI("GetPredictedLatency starts : %d", wid);
  return latency_models_[wid]->Predict(model_id);
}

TfLiteStatus ModelManager::Update(Job& job) {
  thermal_models_[job.worker_id]->Update(job);
  latency_models_[job.worker_id]->Update(job.model_id, job.latency);
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite