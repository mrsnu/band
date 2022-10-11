#include "tensorflow/lite/splash/processor_thermal_model.h"
#include <cerrno>
#include <cassert>
#include <fstream>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/builtin_ops.h"

namespace tflite {
namespace impl {

using namespace std;

TfLiteStatus ProcessorThermalModel::Init(int32_t worker_size) {
  temp_param_.assign(worker_size, vector<double>(worker_size, 1.0));
  freq_param_.assign(worker_size, vector<double>(worker_size, 1.0));
  flops_param_.assign(worker_size, 1.0);
  membytes_param_.assign(worker_size, 1.0);
  error_param_.assign(worker_size, 1.0);
  return kTfLiteOk;
}

vector<thermal_t> ProcessorThermalModel::Predict(const Subgraph* subgraph) {
  // Get temperature

  // Get frequency 

  // Get flops 
  int64_t flops = EstimateFLOPS(subgraph, subgraph);

  // Get membytes 
  int64_t memBytes = EstimateInputOutputSize(subgraph);

  vector<thermal_t> future_temperature;
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


TfLiteStatus ProcessorThermalModel::Update(vector<thermal_t> error) {
  return kTfLiteOk;
}


} // namespace impl
} // namespace tflite