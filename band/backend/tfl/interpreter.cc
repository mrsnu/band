#include "band/backend/tfl/interpreter.h"

#include "band/backend/tfl/model.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend/tfl/util.h"
#include "band/error_reporter.h"
#include "band/logger.h"
#include "tensorflow/lite/context_util.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace Band {
namespace TfLite {
TfLiteInterpreter::~TfLiteInterpreter() {
  // explicitly remove interpreters first
  // since delegates own interpreter.
  interpreters_.clear();
  delegates_.clear();
}

ModelSpec TfLiteInterpreter::InvestigateModelSpec(Interface::IModel* model) {
  ModelSpec model_spec;

  // Analyze entire model based on CPU interpereter
  {
    std::unique_ptr<tflite::Interpreter> interpreter =
        CreateTfLiteInterpreter(model, kBandCPU);

    tflite::impl::Subgraph& primary_subgraph = interpreter->primary_subgraph();
    std::vector<int>& execution_plan = primary_subgraph.execution_plan();
    model_spec.num_ops = execution_plan.size();

    // allocate circular buffer for model IO
    std::vector<TfLiteTensor*> input_tensors;
    std::vector<TfLiteTensor*> output_tensors;

    for (int input_tensor : primary_subgraph.inputs()) {
      input_tensors.push_back(primary_subgraph.tensor(input_tensor));
    }

    for (int output_tensor : primary_subgraph.outputs()) {
      output_tensors.push_back(primary_subgraph.tensor(output_tensor));
    }
    // check input/output/intermediate tensors to fill in
    // model_spec.output_tensors and model_spec.tensor_types
    for (auto node_index : execution_plan) {
      const TfLiteNode& node =
          primary_subgraph.node_and_registration(node_index)->first;

      std::set<int> tensor_indices;
      for (int input_tensor : tflite::TfLiteIntArrayView(node.inputs)) {
        tensor_indices.insert(input_tensor);
      }

      for (int output_tensor : tflite::TfLiteIntArrayView(node.outputs)) {
        tensor_indices.insert(output_tensor);
        model_spec.node_output_tensors.insert(output_tensor);
      }

      for (auto i : tensor_indices) {
        const auto* tensor = primary_subgraph.tensor(i);
        model_spec.tensor_types.insert(GetBandType(tensor->type));
      }
    }

    std::copy(primary_subgraph.inputs().begin(),
              primary_subgraph.inputs().end(),
              std::inserter(model_spec.input_tensors,
                            model_spec.input_tensors.begin()));

    std::copy(primary_subgraph.outputs().begin(),
              primary_subgraph.outputs().end(),
              std::inserter(model_spec.output_tensors,
                            model_spec.output_tensors.begin()));
  }

  // also check unsupported ops to fill in model_spec.unsupported_ops
  for (int i = 0; i < kBandNumDevices; ++i) {
    BandDeviceFlags device_flag = static_cast<BandDeviceFlags>(i);

    if (device_flag == kBandCPU) {
      // no need to check supportability for CPU
      continue;
    }

    std::unique_ptr<tflite::Interpreter> interpreter =
        CreateTfLiteInterpreter(model, device_flag);

    if (!interpreter) {
      continue;
    }

    tflite::impl::Subgraph& primary_subgraph = interpreter->primary_subgraph();
    std::vector<int>& execution_plan = primary_subgraph.execution_plan();

    for (auto node_index : execution_plan) {
      const TfLiteNode& node =
          primary_subgraph.node_and_registration(node_index)->first;
      if (node.delegate == nullptr) {
        // this subgraph is always a 0~num_ops-1 CPU subgraph so
        // the node-->op mapping is basically the identity mapping
        model_spec.unsupported_ops[device_flag].insert(node_index);
      }
    }
  }

  return model_spec;
}

BandStatus TfLiteInterpreter::FromModel(Interface::IModel* model,
                                        WorkerId worker_id,
                                        BandDeviceFlags device,
                                        std::set<int> ops) {
  TfLiteStatus status = kTfLiteOk;
  std::unique_ptr<tflite::Interpreter> interpreter =
      CreateTfLiteInterpreter(model, device, ops);

  if (!interpreter) {
    return kBandError;
  }

  if (worker_id_ == -1) {
    worker_id_ = worker_id;
  }

  if (worker_id != worker_id_) {
    return kBandError;
  }

  // model-level subgraph
  if (ops.size() == 0) {
    interpreters_[SubgraphKey(model->GetId(), worker_id)] =
        std::move(interpreter);
  } else {
    auto vec2set = [](std::vector<int> vec) {
      return std::set<int>(vec.begin(), vec.end());
    };
    interpreters_[SubgraphKey(
        model->GetId(), worker_id, vec2set(interpreter->inputs()),
        vec2set(interpreter->outputs()))] = std::move(interpreter);
  }

  return GetBandStatus(status);
}

BandBackendType TfLiteInterpreter::GetBackendType() const {
  return kBandTfLite;
}

const std::vector<int>& TfLiteInterpreter::GetInputs(
    const SubgraphKey& key) const {
  return GetInterpreter(key)->inputs();
}

const std::vector<int>& TfLiteInterpreter::GetOutputs(
    const SubgraphKey& key) const {
  return GetInterpreter(key)->outputs();
}

const char* TfLiteInterpreter::GetInputName(const SubgraphKey& key,
                                            int index) const {
  return GetInterpreter(key)->GetInputName(index);
}

const char* TfLiteInterpreter::GetOutputName(const SubgraphKey& key,
                                             int index) const {
  return GetInterpreter(key)->GetOutputName(index);
}

size_t TfLiteInterpreter::GetNumTensors(const SubgraphKey& key) const {
  return GetInterpreter(key)->tensors_size();
}

size_t TfLiteInterpreter::GetNumNodes(const SubgraphKey& key) const {
  return GetInterpreter(key)->nodes_size();
}

std::shared_ptr<Interface::ITensorView> TfLiteInterpreter::GetTensorView(
    const SubgraphKey& key, int index) {
  return std::make_shared<TfLiteTensorView>(GetInterpreter(key)->tensor(index));
}

SubgraphKey TfLiteInterpreter::GetModelSubgraphKey(ModelId model_id) const {
  // doesn't need to validate worker id
  // since it creates invalid subgraph key
  return SubgraphKey(model_id, worker_id_);
}

bool TfLiteInterpreter::HasSubgraph(const SubgraphKey& key) const {
  return interpreters_.find(key) != interpreters_.end();
}

BandStatus TfLiteInterpreter::InvokeSubgraph(const SubgraphKey& key) {
  if (!HasSubgraph(key)) {
    return kBandError;
  }
  BandStatus status = GetBandStatus(interpreters_[key]->Invoke());
  BAND_LOG_INTERNAL(BAND_LOG_INFO, "Invoke %d", status);
  return status;
}

tflite::Interpreter* TfLiteInterpreter::GetInterpreter(const SubgraphKey& key) {
  auto it = interpreters_.find(key);
  return it != interpreters_.end() ? it->second.get() : nullptr;
}

const tflite::Interpreter* TfLiteInterpreter::GetInterpreter(
    const SubgraphKey& key) const {
  auto it = interpreters_.find(key);
  return it != interpreters_.end() ? it->second.get() : nullptr;
}

// Discard nnapi backend for devices that has direct support
bool IsNNAPIDeviceUseful(std::string name) {
  static const char* const filter_keywords[] = {
      "nnapi-reference",  // CPU
      "gpu",              // Inefficient than GPUDelegate
      "default"};

  for (auto keyword : filter_keywords) {
    if (name.find(keyword) != std::string::npos) return false;
  }

  return true;
}

BandDeviceFlags GetNNAPIDeviceFlag(std::string name) {
  auto contains_keywords = [&name](std::vector<std::string> keywords) {
    for (auto keyword : keywords) {
      if (name.find(keyword) != std::string::npos) return true;
    }
    return false;
  };

  if (contains_keywords({"gpu"})) {
    return kBandGPU;
  }

  if (contains_keywords({"dsp"})) {
    return kBandDSP;
  }

  if (contains_keywords({
          "google-edgetpu",
          "liteadaptor",  // Huawei (DaVinci NPU)
          "neuron-ann",   // Mediatek APU
          "qti-hta",      // Hexagon tensor accelerator
          "mtk-neuron"    // Mediatek APU
                        // "mtk-mdla" #TODO(#139) - Mediatek APU for half float
      })) {
    return kBandNPU;
  }

  // TODO #23
  // 1. Add additional NPU / TPU names
  // 2. Is 'hta' belongs to dsp or npu?

  return kBandNumDevices;
}

std::unique_ptr<tflite::Interpreter> TfLiteInterpreter::CreateTfLiteInterpreter(
    Interface::IModel* model, BandDeviceFlags device,
    std::set<int> op_indices) {
  // TODO: Build subgraph based on op_indices
  std::unique_ptr<tflite::Interpreter> interpreter;

  TfLiteModel* tf_model = static_cast<TfLiteModel*>(model);
  if (!IsCompatible(model) || !tf_model || !tf_model->IsInitialized()) {
    return nullptr;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::impl::InterpreterBuilder builder(*tf_model->GetFlatBufferModel(),
                                           resolver);
  TfLiteStatus status = builder(&interpreter);

  auto delegate = GetDeviceDelegate(device);

  if (delegate.first == kBandError || !interpreter) {
    return nullptr;
  } else {
  }

  if (device != kBandCPU && GetBandStatus(interpreter->ModifyGraphWithDelegate(
                                delegate.second)) != kBandOk) {
    return nullptr;
  }

  if (GetBandStatus(interpreter->AllocateTensors()) != kBandOk) {
    return nullptr;
  }

  return interpreter;
}

std::pair<BandStatus, TfLiteDelegate*> TfLiteInterpreter::GetDeviceDelegate(
    BandDeviceFlags device) {
  auto delegate_it = delegates_.find(device);
  if (delegate_it != delegates_.end()) {
    return {kBandOk, delegate_it->second.get()};
  } else {
    tflite::Interpreter::TfLiteDelegatePtr target_delegate =
        tflite::Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});

    std::vector<const char*> string_device_names_list;
    switch (device) {
      case kBandCPU: {
        // TODO #23: XNNPACK seems inefficient than default CPU
        // Only valid case to return Ok with nullptr
        return {kBandOk, nullptr};
        break;
      }

#if defined(__ANDROID__)
      case kBandGPU: {
        TfLiteGpuDelegateOptionsV2 gpu_opts =
            TfLiteGpuDelegateOptionsV2Default();
        gpu_opts.inference_priority1 =
            TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        gpu_opts.inference_priority2 =
            TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        gpu_opts.inference_priority3 =
            TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        gpu_opts.experimental_flags |=
            TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

        // set this to a large number so that we can prevent this from getting
        // defaulted to 1 (cf. #34)
        gpu_opts.max_delegated_partitions = 100;
        target_delegate = tflite::Interpreter::TfLiteDelegatePtr(
            TfLiteGpuDelegateV2Create(&gpu_opts), &TfLiteGpuDelegateV2Delete);
        break;
      }

      case kBandDSP:
      case kBandNPU: {
        string_device_names_list = tflite::nnapi::GetDeviceNamesList();

        // TODO #23 : Add more nnapi names
        // Possible device runtime names
        // nnapi : nnapi-default, nnapi-reference
        // armnn : armnn
        // qualcomm : qti-default, qti-gpu, qti-dsp, qti-hta
        // mediatek : neuron-ann, mtk-gpu, mtk-dsp, mtk-neuron, mtk-mdla
        // google tpu : google-edgetpu
        // huawei npu : liteadaptor
        for (const char* device_name : string_device_names_list) {
          if (IsNNAPIDeviceUseful(device_name)) {
            BAND_LOG_INTERNAL(BAND_LOG_INFO, "Available NNAPI device name %s",
                              device_name);
            tflite::StatefulNnApiDelegate::Options nnapi_options =
                tflite::StatefulNnApiDelegate::Options();
            // Unlimited partition : 0
            nnapi_options.max_number_delegated_partitions = 0;
            nnapi_options.accelerator_name = device_name;

            tflite::Interpreter::TfLiteDelegatePtr nnapi_delegate =
                tflite::Interpreter::TfLiteDelegatePtr(
                    new tflite::StatefulNnApiDelegate(nnapi_options),
                    [](TfLiteDelegate* delegate) {
                      delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(
                          delegate);
                    });

            if (nnapi_delegate.get()) {
              TfLiteDelegateFlags delegate_flag =
                  static_cast<TfLiteDelegateFlags>(nnapi_delegate->flags);
              auto nnapi_options = tflite::StatefulNnApiDelegate::GetOptions(
                  nnapi_delegate.get());

              if (device == kBandDSP &&
                  GetNNAPIDeviceFlag(nnapi_options.accelerator_name) ==
                      kBandDSP) {
                target_delegate = std::move(nnapi_delegate);
              }

              if (device == kBandNPU &&
                  GetNNAPIDeviceFlag(nnapi_options.accelerator_name) ==
                      kBandNPU) {
                target_delegate = std::move(nnapi_delegate);
              }
            }
          }
        }

        break;
      }

#endif  // defined(__ANDROID__)

      default: {
        return {kBandError, nullptr};
        break;
      }
    }

    bool success = target_delegate != nullptr;

    if (success) {
      delegates_.insert({device, std::move(target_delegate)});
    }

    return {success ? kBandOk : kBandError, target_delegate.get()};
  }
}  // namespace TfLite

}  // namespace TfLite
}  // namespace Band