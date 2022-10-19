#include "tensorflow/lite/splash/processor_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/QR"
#include "third_party/eigen3/Eigen/Cholesky"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

#if defined(__ANDROID__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "thermal", __VA_ARGS__)
#include <android/log.h>
#endif // defined(__ANDROID__)

namespace tflite {
namespace impl {

using namespace std;
using namespace Eigen;

TfLiteStatus ProcessorThermalModel::Init(int32_t worker_size, int32_t window_size) {
  model_param_.assign(worker_size, vector<double>(8, 1.));
  window_size_ = window_size;
  return kTfLiteOk;
}

vector<thermal_t> ProcessorThermalModel::Predict(const Subgraph* subgraph) {
  LOGI("ProcessorThermalModel::Predict starts");
  vector<int32_t> regressor;
  // Get temperature from resource monitor
  vector<thermal_t> temp = GetResourceMonitor().GetAllTemperature();
  regressor.insert(regressor.end(), temp.begin(), temp.end());

  // Get frequency 
  vector<freq_t> freq = GetResourceMonitor().GetAllTemperature();
  regressor.insert(regressor.end(), freq.begin(), freq.end());

  // Get flops 
  // flops_regressor_ = EstimateFLOPS(subgraph, subgraph) / 100000;

  // Get membytes 
  // membytes_regressor_ = EstimateInputOutputSize(subgraph);

  int32_t expected_latency;
  regressor.push_back(expected_latency);

  // Push error term
  regressor.push_back(1);

  vector<thermal_t> future_temperature;

  if (regressor.size() != model_param_[0].size()) {
    LOGI("[ProcessorThermalModel] Error!!: regressor.size() != model_param_.size()");
    return future_temperature;
  }

  for (int wid = 0; wid < model_param_.size(); wid++) {
    thermal_t predicted = 0;
    for (int i = 0; i < regressor.size(); i++) {
      predicted += regressor[i] * model_param_[wid][i];
    }
    future_temperature[wid] = predicted;
  }
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

void ProcessorThermalModel::PrintParameters() {
  LOGI("================Temp Param(S)================");
  for (auto wid = 0; wid < model_param_.size(); ++wid) {
    std::stringstream ss;
    for (auto i = 0; i < model_param_[wid].size(); ++i) {
      ss << model_param_[wid][i] << '\t';
    }
    LOGI("%s", ss.str().c_str());
  }
  LOGI("================Temp Param(E)================");
}

TfLiteStatus ProcessorThermalModel::Update(Job job) {
  LOGI("ProcessorThermalModel::Update starts");
  // Add job to log table
  ThermalLog log(job);
  if (log_.size() > window_size_) {
    log_.pop_front();
  }
  log_.push_back(log);

  if (log_.size() < 50) {
    LOGI("ProcessorThermalModel::Update Not enough data : %d", log_.size());
    return kTfLiteOk;
  }

  // PrintParameters();

  // Update parameters via normal equation with log table
  for (auto target_worker = 0; target_worker < model_param_.size(); target_worker++) {
    Eigen::MatrixXd X;
    X.resize(log_.size(), 8);
    Eigen::VectorXd Y;
    Y.resize(log_.size(), 1);
    for (auto i = 0; i < log_.size(); i++) {
      // TODO : init matrix with vector more efficient way
      X(i, 0) = log_[i].before_temp[0];
      X(i, 1) = log_[i].before_temp[1];
      X(i, 2) = log_[i].before_temp[2];
      X(i, 3) = log_[i].before_temp[3];
      X(i, 4) = log_[i].frequency[0];
      X(i, 5) = log_[i].frequency[1];
      X(i, 6) = log_[i].latency;
      X(i, 7) = 1.0;
      Y(i, 0) = log_[i].after_temp[target_worker];
    }
    Eigen::Matrix<double, 1, 8> theta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
    for (auto i = 0; i < model_param_[target_worker].size(); i++) {
      model_param_[target_worker][i] = theta(0, i); 
    }
  }
  // PrintParameters();

  LOGI("ProcessorThermalModel::Update Ends");
  return kTfLiteOk;
}

} // namespace impl
} // namespace tflite