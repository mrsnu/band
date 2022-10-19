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

int64_t ModelManager::GetFlops(const Subgraph* subgraph) {
  int64_t flops = 0;
  for (int op_index : subgraph->op_indices()) {
    const auto node_registration =
        subgraph->node_and_registration(op_index);
    const TfLiteNode& node = node_registration->first;
    const TfLiteRegistration& registration = node_registration->second;
    switch (registration.builtin_code) {
      case kTfLiteBuiltinConv2d:
      case kTfLiteBuiltinDepthwiseConv2d: {
        assert(node.inputs->size == 3);
        assert(node.outputs->size == 1);
        const TfLiteTensor* input =
            subgraph->tensor(node.inputs->data[0]);
        const TfLiteTensor* weight =
            subgraph->tensor(node.inputs->data[1]);
        const TfLiteTensor* bias =
            subgraph->tensor(node.inputs->data[2]);
        const TfLiteTensor* output =
            subgraph->tensor(node.outputs->data[0]);
        assert(input->dims->size == 4);   // batch, iw, ih, ic
        assert(weight->dims->size == 4);  // oc, kw, kh, ic
        assert(bias->dims->size == 1);    // oc
        assert(output->dims->size == 4);  // batch, ow, oh, oc

        const int64_t kw = weight->dims->data[1];
        const int64_t kh = weight->dims->data[2];
        const int64_t ic = input->dims->data[3];
        const int64_t oc = output->dims->data[3];
        const int64_t o_size = output->dims->data[0] * output->dims->data[1] *
                               output->dims->data[2];

        int64_t conv_flops = o_size * kw * kh * ic * oc;
        if (registration.builtin_code == kTfLiteBuiltinDepthwiseConv2d) {
          conv_flops /= ic;
        }
        flops += conv_flops;
      } break;
      case kTfLiteBuiltinTransposeConv: {
        assert(node.inputs->size == 3);
        assert(node.outputs->size == 1);
        const TfLiteTensor* bias =
            subgraph->tensor(node.inputs->data[0]);
        const TfLiteTensor* weight =
            subgraph->tensor(node.inputs->data[1]);
        const TfLiteTensor* input =
            subgraph->tensor(node.inputs->data[2]);
        const TfLiteTensor* output =
            subgraph->tensor(node.outputs->data[0]);
        assert(bias->dims->size == 1);    // ??
        assert(weight->dims->size == 4);  // oc, kw, kh, ic
        assert(input->dims->size == 4);   // batch, iw, ih, ic
        assert(output->dims->size == 4);  // batch, ow, oh, oc

        const int64_t kw = weight->dims->data[1];
        const int64_t kh = weight->dims->data[2];
        const int64_t ic = input->dims->data[3];
        const int64_t oc = output->dims->data[3];
        const int64_t i_size =
            input->dims->data[0] * input->dims->data[1] * input->dims->data[2];

        int64_t trconv_flops = i_size * kw * kh * ic * oc;
        flops += trconv_flops;
      } break;
      default:
        break;
    }
  }
  return flops;
}

int64_t ModelManager::GetMembytes(const Subgraph* subgraph) {
  // TODO: Add input/output tensors without weights.
  const std::vector<int>& input_tensors = subgraph->inputs();
  const std::vector<int>& output_tensors = subgraph->outputs();
  int64_t subgraph_input_output_size = 0;
  for (int tensor_idx : input_tensors) {
    subgraph_input_output_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  for (int tensor_idx : output_tensors) {
    subgraph_input_output_size += (int64_t)subgraph->tensor(tensor_idx)->bytes;
  }
  return subgraph_input_output_size;
}


} // namespace impl
} // namespace tflite