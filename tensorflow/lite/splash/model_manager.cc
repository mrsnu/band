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

bool ModelManager::IsAvailableWorker(worker_id_t wid, Subgraph* subgraph, 
                                     std::vector<thermal_t> before_temp) {
  auto& thermal_model = thermal_models_[wid]; 
  int64_t latency = GetPredictedLatency(wid, subgraph->GetKey().model_id);
  std::vector<thermal_t> temperature = thermal_model->Predict(subgraph, latency, before_temp); 
  for (int i = 0; i < temperature.size(); i++) {
    thermal_t temp = temperature[i];
    auto threshold = resource_monitor_.GetThrottlingThreshold(i);
    if (temp > threshold) {
      LOGI("Throttling predicted : [worker %d]'s temperature[%d] will exceeds threshold[%d] on assigning to worker [%d]", i, temp, threshold, wid);
      return false;
    }
  }
  return true;
}

std::vector<thermal_t> ModelManager::GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph, std::vector<thermal_t> before_temp) {
  auto latency = GetPredictedLatency(wid, subgraph->GetKey().model_id);
  return thermal_models_[wid]->Predict(subgraph, latency, before_temp);
}

int64_t ModelManager::GetPredictedLatency(worker_id_t wid, int32_t model_id) {
  return latency_models_[wid]->Predict(model_id);
}

int64_t ModelManager::GetPredictedThrottledLatency(worker_id_t wid, int32_t model_id) {
  return latency_models_[wid]->PredictThrottled(model_id);
}

TfLiteStatus ModelManager::Update(Job& job) {
  thermal_models_[job.worker_id]->Update(job);
  latency_models_[job.worker_id]->Update(job.model_id, job.latency);
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite