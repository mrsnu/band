#include "tensorflow/lite/splash/processor_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "thermal", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus ProcessorThermalModel::Init(int32_t worker_size) {
  temp_param_.assign(worker_size, vector<double>(worker_size, 0.2));
  freq_param_.assign(worker_size, vector<double>(worker_size, 0.000001));
  flops_param_.assign(worker_size, 0.01);
  membytes_param_.assign(worker_size, 0.00001);
  error_param_.assign(worker_size, 1.0);
  return kTfLiteOk;
}

vector<thermal_t> ProcessorThermalModel::Predict(const Subgraph* subgraph) {
  LOGI("ProcessorThermalModel::Predict starts");
  // Get temperature from resource monitor
  temp_regressor_ = GetResourceMonitor().GetAllTemperature();

  // Get frequency 
  freq_regressor_ = GetResourceMonitor().GetAllFrequency();

  // Get flops 
  flops_regressor_ = EstimateFLOPS(subgraph, subgraph) / 100000;

  LOGI("Flops : %lld", flops_regressor_);
  // Get membytes 
  membytes_regressor_ = EstimateInputOutputSize(subgraph);

  LOGI("memBytes : %lld", membytes_regressor_);
  return EstimateFutureTemperature(temp_regressor_, freq_regressor_, flops_regressor_, membytes_regressor_);
}

vector<thermal_t> ProcessorThermalModel::EstimateFutureTemperature(const vector<thermal_t> temp,
                                                                   const vector<freq_t> freq,
                                                                   const int64_t flops,
                                                                   const int64_t membytes) {
  vector<thermal_t> future_temperature;
  // TODO: Refactor this calculation
  future_temperature = Plus(Plus(Multiply(temp_param_, temp), Multiply(freq_param_, freq)), 
    Plus(Plus(Multiply(flops_param_, flops), Multiply(membytes_param_, membytes)), Multiply(error_param_, 1)));
  return future_temperature;
}

int64_t ProcessorThermalModel::EstimateFLOPS(const Subgraph* subgraph,
                                             const Subgraph* primary_subgraph) {
  int64_t flops = 0;
  for (int op_index : subgraph->op_indices()) {
    const auto node_registration =
        primary_subgraph->node_and_registration(op_index);
    const TfLiteNode& node = node_registration->first;
    const TfLiteRegistration& registration = node_registration->second;
    switch (registration.builtin_code) {
      case kTfLiteBuiltinConv2d:
      case kTfLiteBuiltinDepthwiseConv2d: {
        assert(node.inputs->size == 3);
        assert(node.outputs->size == 1);
        const TfLiteTensor* input =
            primary_subgraph->tensor(node.inputs->data[0]);
        const TfLiteTensor* weight =
            primary_subgraph->tensor(node.inputs->data[1]);
        const TfLiteTensor* bias =
            primary_subgraph->tensor(node.inputs->data[2]);
        const TfLiteTensor* output =
            primary_subgraph->tensor(node.outputs->data[0]);
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
            primary_subgraph->tensor(node.inputs->data[0]);
        const TfLiteTensor* weight =
            primary_subgraph->tensor(node.inputs->data[1]);
        const TfLiteTensor* input =
            primary_subgraph->tensor(node.inputs->data[2]);
        const TfLiteTensor* output =
            primary_subgraph->tensor(node.outputs->data[0]);
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

int64_t ProcessorThermalModel::EstimateInputOutputSize(const Subgraph* subgraph) {
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

void ProcessorThermalModel::UpdateParameters(vector<vector<double>>& params, 
                                           vector<thermal_t> error,
                                           vector<thermal_t> regressor) {
  const int error_rows = error.size();
  const int reg_rows = regressor.size();

  for (auto i = 0; i < error_rows; ++i) {
    for (auto k = 0; k < reg_rows; ++k) {
        params[i][k] += (double) (error[i] * regressor[k] * gain_);
    }
  }
}

void ProcessorThermalModel::UpdateParameters(vector<double>& params, 
                                           vector<thermal_t> error,
                                           int64_t regressor) {
  const int error_rows = error.size();
  for (auto i = 0; i < error_rows; ++i) {
    params[i] += (double) (error[i] * regressor * gain_);
  }
}

void ProcessorThermalModel::PrintParameters() {
  LOGI("================Temp Param(S)================");
  for (auto i = 0; i < temp_param_.size(); ++i) {
    LOGI("%.4f\t%.4f\t%.4f\t%.4f\t%.4f", 
        temp_param_[i][0], temp_param_[i][1], temp_param_[i][2], temp_param_[i][3], temp_param_[i][4]);
  }
  LOGI("================Temp Param(E)================");
}


TfLiteStatus ProcessorThermalModel::Update(vector<thermal_t> error) {
  LOGI("ProcessorThermalModel::Update starts");
  for (int i = 0; i < kTfLiteNumDevices; i++) {
    LOGI("Error[%d] = %d", i, error[i]);
  }
  PrintParameters();

  // Calculate gain first
  UpdateParameters(temp_param_, error, temp_regressor_);
  UpdateParameters(freq_param_, error, freq_regressor_);
  UpdateParameters(flops_param_, error, flops_regressor_);
  UpdateParameters(membytes_param_, error, membytes_regressor_);
  UpdateParameters(error_param_, error, 1);
  LOGI("ProcessorThermalModel::Update Ends");
  PrintParameters();

  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite