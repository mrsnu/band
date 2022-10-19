#include "tensorflow/lite/splash/thermal_model_manager.h"

#include "tensorflow/lite/splash/thermal_model.h"
#include "tensorflow/lite/splash/cloud_thermal_model.h"
#include "tensorflow/lite/splash/processor_thermal_model.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {


TfLiteStatus ThermalModelManager::Init() {
  LOGI("ThermalModelManager:: init");
  for (int wid = 0; wid < kTfLiteNumDevices; wid++) {
    std::unique_ptr<IThermalModel> model = BuildModel(wid);
    models_.emplace_back(std::move(model));
  }
  for (auto& model : models_) {
    auto status = model->Init(models_.size());
    if (status == kTfLiteError) {
      return kTfLiteError;
    }
  }
  LOGI("ThermalModelManager:: finish");
  return kTfLiteOk;
}

std::unique_ptr<IThermalModel> ThermalModelManager::BuildModel(worker_id_t wid) {
  switch (wid) {
    case kTfLiteCPU:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteGPU:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    // case kTfLiteDSP:
    //   return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteNPU:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
    case kTfLiteCLOUD:
      return std::make_unique<CloudThermalModel>(wid, resource_monitor_);
    default:
      return std::make_unique<ProcessorThermalModel>(wid, resource_monitor_);
  }
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

TfLiteStatus ThermalModelManager::Update(Job& job) {
  std::vector<thermal_t> error(kTfLiteNumDevices, 0);
  for (int i = 0; i < kTfLiteNumDevices; i++) {
    LOGI("real_temp = %d, estimated_temp = %d", job.real_temp[i], job.estimated_temp[i]);
    error[i] = job.real_temp[i] - job.estimated_temp[i];
  }
  return models_[job.worker_id]->Update(error);
}

int64_t ThermalModelManager::GetFlops(const Subgraph* subgraph) {
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

int64_t ThermalModelManager::GetMembytes(const Subgraph* subgraph) {
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