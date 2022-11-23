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
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {


TfLiteStatus ModelManager::Init(ResourceConfig& config, bool is_thermal_aware) {
  LOGI("ModelManager:: init");
  // Build ThermalModel
  for (int wid = 0; wid < kTfLiteNumDevices; wid++) {
    std::unique_ptr<IThermalModel> model = BuildThermalModel(wid);
    thermal_models_.emplace_back(std::move(model));
  }
  for (auto& thermal_model : thermal_models_) {
    auto status = thermal_model->Init(config);
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }

  // Build LatencyModel
  for (int wid = 0; wid < kTfLiteNumDevices; wid++) {
    std::unique_ptr<ILatencyModel> model = BuildLatencyModel(wid, is_thermal_aware);
    latency_models_.push_back(std::move(model));
  }
  for (auto& latency_model : latency_models_) {
    auto status = latency_model->Init(config);
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

std::unique_ptr<IThermalModel> ModelManager::BuildThermalModel(worker_id_t wid) {
  if (wid == kTfLiteCLOUD) {
    return std::make_unique<CloudThermalModel>(wid, resource_monitor_);
  }
  return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
}

std::unique_ptr<ILatencyModel> ModelManager::BuildLatencyModel(worker_id_t wid, bool is_thermal_aware) {
  if (wid == kTfLiteCLOUD) {
    return std::make_unique<CloudLatencyModel>(wid, resource_monitor_, is_thermal_aware);
  }
  return std::make_unique<ProcessorLatencyModel>(wid, resource_monitor_, is_thermal_aware);
}

bool ModelManager::IsAvailableWorker(worker_id_t wid, Subgraph* subgraph) {
  auto& thermal_model = thermal_models_[wid]; 
  std::vector<thermal_t> before_temp = resource_monitor_.GetAllTemperature();
  int64_t latency = GetPredictedLatency(wid, subgraph);
  thermal_t temp = thermal_model->Predict(subgraph, latency, before_temp); 
  auto threshold = resource_monitor_.GetThrottlingThreshold(wid);
  if (temp > threshold) {
    // LOGI("Throttling predicted : [worker %d]'s temp[%d] will exceeds threshold[%d]", wid, temp, threshold);
    return false;
  }
  thermal_t target_temp = thermal_model->PredictTarget(subgraph, latency, before_temp); 
  auto target_threshold = resource_monitor_.GetTargetThreshold(wid);
  if (target_temp > target_threshold) {
    // LOGI("Target Throttling predicted : [worker %d]'s target temp[%d] will exceeds threshold[%d]", wid, target_temp, target_threshold);
    return false;
  }
  return true;
}

thermal_t ModelManager::GetPredictedTemperature(worker_id_t wid, Subgraph* subgraph) {
  std::vector<thermal_t> before_temp = resource_monitor_.GetAllTemperature();
  auto latency = GetPredictedLatency(wid, subgraph);
  return thermal_models_[wid]->PredictTarget(subgraph, latency, before_temp);
}

std::pair<thermal_t, int64_t>
ModelManager::GetPredictedTempAndLatency(worker_id_t wid, Subgraph* subgraph) {
  std::vector<thermal_t> before_temp = resource_monitor_.GetAllTargetTemperature();
  auto latency = GetPredictedLatency(wid, subgraph);
  auto future_temp = thermal_models_[wid]->PredictTarget(subgraph, latency, before_temp);
  thermal_t temp_diff = future_temp - before_temp[wid];
  return { std::max(0, temp_diff), latency };
}

int64_t ModelManager::GetPredictedLatency(worker_id_t wid, Subgraph* subgraph) {
  return latency_models_[wid]->Predict(subgraph);
}

TfLiteStatus ModelManager::Update(Job& job, Subgraph* subgraph) {
  thermal_models_[job.worker_id]->Update(job, subgraph);
  latency_models_[job.worker_id]->Update(job, subgraph);
  return kTfLiteOk;
}

TfLiteStatus ModelManager::ProfileLatency(Subgraph* subgraph, int64_t latency) {
  Job job = Job(subgraph->GetKey().model_id);
  job.latency = latency;
  latency_models_[subgraph->GetKey().worker_id]->Update(job, subgraph);
  return kTfLiteOk;
}

TfLiteStatus ModelManager::Close() {
  for (auto& thermal_model : thermal_models_) {
    auto status = thermal_model->Close();
    if (status == kTfLiteError) {
      LOGI("Thermal model Error = %d", thermal_model->GetWorkerId());
      return kTfLiteError;
    }
  }
  for (auto& latency_model : latency_models_) {
    auto status = latency_model->Close();
    if (status == kTfLiteError) {
      LOGI("Latency model Error = %d", latency_model->GetWorkerId());
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite