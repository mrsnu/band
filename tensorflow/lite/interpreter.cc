/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/interpreter.h"

#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <utility>
#include <list>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/delegates/status.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tflite_with_xnnpack_optional.h"
#ifdef TFLITE_BUILD_WITH_XNNPACK_DELEGATE
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif
#include "tensorflow/lite/profiling/time_profiler.h"
#include "tensorflow/lite/profiling/time.h"

// TODO(b/139446230): Move to portable platform header.
#if defined(__ANDROID__)
#include "tensorflow/lite/nnapi/nnapi_util.h"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "libtflite", __VA_ARGS__)
#include <android/log.h>
#define TFLITE_IS_MOBILE_PLATFORM
#endif  // defined(__ANDROID__)

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define TFLITE_IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define TFLITE_IS_MOBILE_PLATFORM
#endif
#endif  // defined(__APPLE__)

// TODO(b/132087118): move static_assert to c_api_internal when compiled with
// C++.
static_assert(sizeof(TfLiteFloat16) == sizeof(uint16_t),
              "Float 16 type must be 16 bits.");

namespace tflite {

namespace impl {

namespace {

// Gets the current TfLiteQuantization from the legacy TfLiteQuantizationParams.
TfLiteQuantization GetQuantizationFromLegacy(
    const TfLiteQuantizationParams& legacy_quantization) {
  TfLiteQuantization quantization;
  quantization.type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(1);
  affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  affine_quantization->scale->data[0] = legacy_quantization.scale;
  affine_quantization->zero_point->data[0] = legacy_quantization.zero_point;
  quantization.params = affine_quantization;

  return quantization;
}

// Discard nnapi backend for devices that has direct support
bool IsNNAPIDeviceUseful(std::string name) {
  static const char* const filter_keywords[] = {
    "nnapi-reference",  // CPU
    "gpu",  // Inefficient than GPUDelegate
    "default"};

  for (auto keyword : filter_keywords) {
    if (name.find(keyword) != std::string::npos) 
      return false;
  }

  return true;
}

// TODO(b/153131797): We have put 'delegate_status' to 0 in the following macro
// temporarily because delegate-specific error codes are either not retrievable
// at the moment, which we will add later.
#define TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(runtime_event, a) \
  do {                                                                      \
    TfLiteStatus status = (a);                                              \
    runtime_event.set_runtime_status(/*delegate_status=*/0,                 \
                                     static_cast<int64_t>(status));         \
    TF_LITE_ENSURE_STATUS(status);                                          \
  } while (0)

}  // namespace

Interpreter::Interpreter(ErrorReporter* error_reporter,
                         RuntimeConfig runtime_config)
    : error_reporter_(error_reporter ? error_reporter :
                                       DefaultErrorReporter()) {
  // TODO(b/128420794): Include the TFLite runtime version in the log.
  // Prod logging is useful for mobile platforms where scraping console logs is
  // critical for debugging.
  TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");

  // Reserve some space for the tensors to avoid excessive resizing.
  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  // This operation is cheap because we allocate the CPU context resources (i.e.
  // threads) lazily.
  own_external_cpu_backend_context_.reset(new ExternalCpuBackendContext());
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();
  
  // Initialize internal backend context for cpu contexts
  own_external_cpu_backend_context_->
      set_internal_backend_context(
          std::make_unique<CpuBackendContext>());

  // Create a Planner instance.
  planner_.reset(new Planner(this, runtime_config.resource_config));
  if (planner_->Init(runtime_config.planner_config) != kTfLiteOk) {
    error_reporter_->Report("Planner::Init() failed.");
    exit(-1);
  }

  // Initialize configurations.
  if (Init(runtime_config.interpreter_config) != kTfLiteOk) {
    error_reporter_->Report("Interpreter::Init() failed.");
    exit(-1);
  }

  std::set<TfLiteDeviceFlags> valid_devices = { kTfLiteCPU, kTfLiteCLOUD };
  // Create Delegates for each device.
  // TODO #13: Create mobile device independent delegate instances
  TfLiteDelegatePtr null_delegate =
      TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  delegates_.insert(std::make_pair(kTfLiteDelegateFlagsNone, std::move(null_delegate)));

#if defined(__ANDROID__)
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
  gpu_opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
  gpu_opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

  // set this to a large number so that we can prevent this from getting
  // defaulted to 1 (cf. #34)
  gpu_opts.max_delegated_partitions = 100;
  TfLiteDelegatePtr gpu_delegate = TfLiteDelegatePtr(
    TfLiteGpuDelegateV2Create(&gpu_opts), &TfLiteGpuDelegateV2Delete);
  if (gpu_delegate.get()) {
    delegates_.insert(std::make_pair(kTfLiteDelegateFlagsGPU, std::move(gpu_delegate)));
    valid_devices.insert(kTfLiteGPU);
  }

  std::vector<const char*> string_device_names_list = nnapi::GetDeviceNamesList();

  // TODO #23 : Add more nnapi names
  // Possible device runtime names 
  // nnapi : nnapi-default, nnapi-reference
  // armnn : armnn
  // qualcomm : qti-default, qti-gpu, qti-dsp, qti-hta
  // mediatek : neuron-ann, mtk-gpu, mtk-dsp, mtk-neuron, mtk-mdla
  // google tpu : google-edgetpu
  // huawei npu : liteadaptor
  // samsung npu : eden-drv
  for (const char* device_name : string_device_names_list) {
    if (IsNNAPIDeviceUseful(device_name)) {
      TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO, "Available NNAPI device name %s", device_name);
      StatefulNnApiDelegate::Options nnapi_options = StatefulNnApiDelegate::Options();
      // Unlimited partition : 0
      nnapi_options.max_number_delegated_partitions = 0;
      nnapi_options.accelerator_name = device_name;

      TfLiteDelegatePtr nnapi_delegate = TfLiteDelegatePtr(
        new StatefulNnApiDelegate(nnapi_options),
          [](TfLiteDelegate* delegate) {
            delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
          });

      if (nnapi_delegate.get()) {
        TfLiteDelegateFlags delegate_flag =
            static_cast<TfLiteDelegateFlags>(nnapi_delegate->flags);
        
        switch (delegate_flag) {
          case kTfLiteDelegateFlagsNNAPIDSP:
            valid_devices.insert(kTfLiteDSP);
            break;
          case kTfLiteDelegateFlagsNNAPINPU:
            valid_devices.insert(kTfLiteNPU);
            break;
          default:
            continue;
        }

        delegates_.insert(
          std::make_pair(delegate_flag, std::move(nnapi_delegate)));
      }
    }
  }

  TfLiteDelegatePtr xnnpack_delegate = MaybeCreateXNNPACKDelegate(1);
  if (xnnpack_delegate.get()) {
    delegates_.insert(std::make_pair(kTfLiteDelegateFlagsXNNPACK, std::move(xnnpack_delegate)));
  }

  // TODO #23
  // Add flex delegate?
  
#endif  // defined(__ANDROID__)

  // Create workers.
  auto& potential_workers = runtime_config.worker_config.workers;
  for (int i = 0; i < potential_workers.size(); i++) {
    TfLiteDeviceFlags device_flag = potential_workers[i];
    if (valid_devices.find(device_flag) != valid_devices.end()) {
      std::unique_ptr<Worker> worker;

      if (device_flag == kTfLiteCLOUD) {
        // continue;
        if (planner_->GetWorkerType() == kGlobalQueue) {
          worker = std::make_unique<GlobalQueueOffloadingWorker>(planner_, device_flag);
        } else {
          worker = std::make_unique<DeviceQueueOffloadingWorker>(planner_, device_flag);
        }
      } else {
        if (planner_->GetWorkerType() == kGlobalQueue) {
          worker = std::make_unique<GlobalQueueWorker>(planner_, device_flag);
        } else {
          worker = std::make_unique<DeviceQueueWorker>(planner_, device_flag);
        }
      }

      if (worker->Init(runtime_config.worker_config, workers_.size()) != kTfLiteOk) {
        LOGI("Worker::Init() failed for worker : %s.", TfLiteDeviceGetName(device_flag));
        exit(-1);
      }
      workers_.emplace_back(std::move(worker));
    } else {
      LOGI("%s worker is not created.",
                 TfLiteDeviceGetName(device_flag));
    }
  }
}

Interpreter::~Interpreter() {
  // The owned external Cpu Backend Context will go out of scope with this
  // interpreter. If we have an external backend context that is not
  // owned, we need to clear the cache for other interpreters that may
  // use the context.
  if (external_contexts_[kTfLiteCpuBackendContext] &&
      (external_contexts_[kTfLiteCpuBackendContext] !=
       own_external_cpu_backend_context_.get())) {
    ExternalCpuBackendContext* external_context =
        static_cast<ExternalCpuBackendContext*>(
            external_contexts_[kTfLiteCpuBackendContext]);
    TfLiteInternalBackendContext* internal_context =
        external_context->internal_backend_context();
    if (internal_context) {
      // This call may have negative performance impacts on the next inference
      // for any interpreter using this context. The cache will be refreshed
      // by the next inference.
      internal_context->ClearCaches();
    }
  }

  // update the profile file to include all new profile results from this run
  profiling::util::UpdateDatabase(profile_database_, model_configs_,
                                  profile_database_json_);
  WriteJsonObjectToFile(profile_database_json_, profile_data_path_);
}


TfLiteStatus Interpreter::Init(InterpreterConfig& config) {
  profile_smoothing_factor_ = config.profile_smoothing_factor;
  subgraph_preparation_type_ = config.subgraph_preparation_type;
  minimum_subgraph_size_ = config.minimum_subgraph_size;

  if (NeedProfile()) {
    profile_data_path_ = config.profile_data_path;
    profile_database_json_ = LoadJsonObjectFromFile(config.profile_data_path);
    // we cannot convert the model name strings to integer ids yet,
    // (profile_database_json_ --> profile_database_)
    // since we don't have anything in model_configs_ at the moment

    // Set how many runs are required to get the profile results.
    profile_online_ = config.profile_config.online;
    profile_num_warmups_ = config.profile_config.num_warmups;
    profile_num_runs_ = config.profile_config.num_runs;
    profile_copy_computation_ratio_ = config.profile_config.copy_computation_ratio;

    TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
               "Set Profiling Configuration: warmup=%d count=%d.",
               profile_num_warmups_, profile_num_runs_);
  }

  const TfLiteCPUMaskFlags cpu_mask = 
      static_cast<TfLiteCPUMaskFlags>(config.cpu_masks);
  auto cpu_mask_set = TfLiteCPUMaskGetSet(cpu_mask);

  LOGI("Set affinity to %s cores.", TfLiteCPUMaskGetName(cpu_mask));

  return SetCPUThreadAffinity(cpu_mask_set);
}

void Interpreter::SetExternalContext(TfLiteExternalContextType type,
                                     TfLiteExternalContext* ctx) {
  if (ctx == own_external_cpu_backend_context_.get()) {
    error_reporter_->Report(
        "WARNING: The passed external context is identical to the internally "
        "owned one.");
    return;
  }

  // We have an internally owned external context of kTfLiteCpuBackendContext.
  // If it's overwritten here, we will release the resource of the internally
  // owned external context.
  // Note: the 'max thread count' info associated with the overwritten context
  // will be lost here, and such info is now determined by the new context, thus
  // affecting how much parallelism a TFLite op would have.
  if (kTfLiteCpuBackendContext == type &&
      external_contexts_[kTfLiteCpuBackendContext] ==
          own_external_cpu_backend_context_.get()) {
    own_external_cpu_backend_context_.reset();
  }

  // Update all subgraph's external context since interpreter owns external contexts
  for (int i = 0; i < subgraphs_size(); i++) {
    subgraph(i)->SetExternalContext(type, ctx);
  }
}

TfLiteStatus Interpreter::SetInputs(size_t subgraph_index, std::vector<int> inputs) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->SetInputs(std::move(inputs));
}

TfLiteStatus Interpreter::SetOutputs(size_t subgraph_index, std::vector<int> outputs) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->SetOutputs(std::move(outputs));
}

TfLiteStatus Interpreter::SetVariables(size_t subgraph_index, std::vector<int> variables) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->SetVariables(std::move(variables));
}

TfLiteStatus Interpreter::AllocateTensors() {
  TfLiteStatus status;

  for (int i = 0; i < subgraphs_.size(); ++i) {
    status = subgraphs_[i]->AllocateTensors();
    if (status != kTfLiteOk)
      return status;
  }

  return kTfLiteOk;
}

TfLiteStatus Interpreter::AllocateTensors(size_t subgraph_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraphs_[subgraph_index]->AllocateTensors();
}

void Interpreter::ReserveNodes(size_t subgraph_index, int count) {
  if (subgraph(subgraph_index))
    subgraph(subgraph_index)->ReserveNodes(count);
}

int Interpreter::AddSubgraph(std::unique_ptr<Subgraph> subgraph) {
  int index = GetSubgraphIdx(subgraph->GetKey());
  if (index == -1) {
    index = subgraphs_.size();
    subgraph_idx_map_[subgraph->GetKey()] = index;
    subgraph->SetProfiler(installed_profiler_, index);
    subgraphs_.emplace_back(std::move(subgraph));
  }
  return index;
}

std::unique_ptr<Subgraph> Interpreter::CreateSubgraph() {
  return std::make_unique<Subgraph>(error_reporter_, external_contexts_,
                                      &subgraphs_, &resources_);
}

void Interpreter::DeleteSubgraphs(size_t starting_index_to_delete,
                                  int subgraphs_to_delete) {
  if (subgraphs_to_delete < 0)
    subgraphs_to_delete = subgraphs_.size() - starting_index_to_delete;

  if (starting_index_to_delete + subgraphs_to_delete <= subgraphs_.size()) {
    subgraphs_.erase(subgraphs_.begin() + starting_index_to_delete,
    subgraphs_.begin() + starting_index_to_delete + subgraphs_to_delete);
  }
}

TfLiteStatus Interpreter::AddNodeWithParameters(
    size_t subgraph_index,
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->AddNodeWithParameters(
      inputs, outputs, {}, init_data, init_data_size, builtin_data,
      registration, node_index);
}

TfLiteStatus Interpreter::ResizeInputTensor(size_t subgraph_index, size_t tensor_index,
                                            const std::vector<int>& dims) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Interpreter::ResizeInputTensorStrict(size_t subgraph_index,
    size_t tensor_index, const std::vector<int>& dims) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->ResizeInputTensorStrict(tensor_index, dims);
}

TfLiteStatus Interpreter::ReleaseNonPersistentMemory(size_t subgraph_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->ReleaseNonPersistentMemory();
}

TfLiteStatus Interpreter::Invoke(size_t subgraph_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(installed_profiler_,
                                                           "invoke");
  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
      scoped_runtime_event, (*subgraphs_[subgraph_index]).Invoke());

  if (!allow_buffer_handle_output_) {
    for (size_t tensor_index : (*subgraphs_[subgraph_index]).outputs()) {
      TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
          scoped_runtime_event,
          (*subgraphs_[subgraph_index]).EnsureTensorDataIsReadable(tensor_index));
    }
  }

  return kTfLiteOk;
}

int Interpreter::InvokeModelAsync(int model_id, Tensors inputs) {
  return InvokeModelAsync(Job(model_id), inputs);
}
 
int Interpreter::InvokeModelAsync(Job request, Tensors inputs) {
  std::vector<int> job_ids = InvokeModelsAsync({request}, {inputs});
  return job_ids.size() == 1 ? job_ids[0] : -1;
}

std::vector<int> Interpreter::InvokeModelsAsync(
    std::vector<Tensors> model_inputs) {
  if (model_inputs.size() != model_configs_.size()) {
    error_reporter_->Report(
        "Invalid input size model_input.size() %d != model_configs_.size() %d.",
        model_inputs.size(), model_configs_.size());
    return {};
  }

  std::vector<Job> requests;
  std::vector<Tensors> request_inputs;
  for (auto& m : model_configs_) {
    int model_id = m.first;
    ModelConfig& model_config = m.second;
    Job request = Job(model_id);
    request.model_fname = model_config.model_fname;
    request.device_id = model_config.device;
    request.slo_us = model_config.slo_us;
    for (int k = 0; k < model_config.batch_size; ++k) {
      requests.push_back(request);
      request_inputs.push_back(model_inputs[model_id]);
    }
  }

  return InvokeModelsAsync(requests, request_inputs);
}

std::vector<int> Interpreter::InvokeModelsAsync(
    std::vector<Job> requests, std::vector<Tensors> request_inputs) {
  for (auto& request : requests) {
    int model_id = request.model_id;
    ModelConfig& model_config = model_configs_[model_id];
    request.model_fname = model_config.model_fname;
    request.device_id = model_config.device;
  }

  if (request_inputs.size() > 0) {
    if (requests.size() != request_inputs.size()) {
      error_reporter_->Report(
          "Invalid input size requests.size() %d != request_inputs.size() %d.",
          requests.size(), request_inputs.size());
      return {};
    }

    for (size_t i = 0; i < requests.size(); i++) {
      Job& request = requests[i];
      int input_handle = model_input_buffer_[request.model_id]->Alloc();
      if (model_input_buffer_[request.model_id]->PutTensorsToHandle(
              request_inputs[i], input_handle) == kTfLiteOk) {
        request.input_handle = input_handle;
        request.output_handle = model_output_buffer_[request.model_id]->Alloc();
      } else {
        error_reporter_->Report("Input copy failure for model %d request %d.",
                                request.model_id, i);
        return {};
      }
    }
  }

  return planner_->EnqueueBatch(requests);
}

void Interpreter::InvokeModelSync(int model_id, Tensors inputs, Tensors outputs) {
  return InvokeModelSync(Job(model_id), inputs, outputs);
}

void Interpreter::InvokeModelSync(Job request, Tensors inputs, Tensors outputs) {
  InvokeModelsSync({request}, {inputs}, {outputs});
}

void Interpreter::InvokeModelsSync(std::vector<Tensors> model_inputs,
                                   std::vector<Tensors> model_outputs) {
  if (model_inputs.size() != model_configs_.size() ||
      model_outputs.size() != model_configs_.size()) {
    error_reporter_->Report(
        "Invalid input/output size model_inputs.size() %d, "
        "model_outputs.size() %d, model_configs_.size() %d.",
        model_inputs.size(), model_outputs.size(), model_configs_.size());
    return;
  }

  std::vector<int> job_ids = InvokeModelsAsync(model_inputs);
  planner_->Wait(job_ids);

  int job_index = 0;
  for (int model_id = 0; model_id < model_configs_.size(); model_id++) {
    const ModelConfig& m = model_configs_[model_id];
    for (int batch_idx = 0; batch_idx < m.batch_size; batch_idx++) {
      GetOutputTensors(job_ids[job_index], model_outputs[model_id]);
      job_index++;
    }
  }
}

void Interpreter::InvokeModelsSync(std::vector<Job> requests,
                                   std::vector<Tensors> request_inputs,
                                   std::vector<Tensors> request_outputs) {
  if (request_inputs.size() > 0 &&
      (request_inputs.size() != requests.size() ||
       request_outputs.size() != requests.size())) {
    error_reporter_->Report(
        "Invalid input/output size request_inputs.size() %d, "
        "request_outputs.size() %d, requests.size() %d.",
        request_inputs.size(), request_outputs.size(), requests.size());
    return;
  }

  std::vector<int> job_ids = InvokeModelsAsync(requests, request_inputs);
  planner_->Wait(job_ids);

  // We don't have to check request_outputs.size() again.
  if (request_inputs.size() > 0) {
    for (size_t i = 0; i < job_ids.size(); i++) {
      GetOutputTensors(job_ids[i], request_outputs[i]);
    }
  }
}

TfLiteStatus Interpreter::GetOutputTensors(int job_id, Tensors& outputs) const {
  Job job = planner_->GetFinishedJob(job_id);

  if (job.job_id == -1) {
    // Not finished yet.
    return kTfLiteOk;
  }

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    error_reporter_->Report("Invalid model_id : %d.", job.model_id);
    return kTfLiteError;
  }

  return model_output_buffer_.at(job.model_id)
      ->GetTensorsFromHandle(outputs, job.output_handle);
}

void Interpreter::SetEndInvokeFunction(
    std::function<void(int, TfLiteStatus)> on_end_invoke) {
  planner_->SetEndInvokeFunction(on_end_invoke);
}

TfLiteStatus Interpreter::AddTensors(size_t subgraph_index, int tensors_to_add,
                                     int* first_new_tensor_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::ResetVariableTensors(size_t subgraph_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->ResetVariableTensors();
}

const char* Interpreter::OpProfilingString(size_t subgraph_index,
                                           const TfLiteRegistration& op_reg,
                                           const TfLiteNode* node) const {
  if (subgraph_index < subgraphs_size() && op_reg.profiling_string) {
    return op_reg.profiling_string(&subgraphs_[subgraph_index]->context_, node);
  }
  return nullptr;
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->SetTensorParametersReadOnly(
      tensor_index, type, name, dims.size(), dims.data(), quantization, buffer,
      bytes, allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    bool is_variable) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->SetTensorParametersReadWrite(
      tensor_index, type, name, dims.size(), dims.data(), quantization,
      is_variable);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name,
    const size_t rank, const int* dims, TfLiteQuantizationParams quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return subgraph(subgraph_index)->SetTensorParametersReadOnly(
      tensor_index, type, name, rank, dims, new_quantization, buffer, bytes,
      allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    size_t subgraph_index, size_t tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return subgraph(subgraph_index)->SetTensorParametersReadWrite(
      tensor_index, type, name, rank, dims, new_quantization, is_variable,
      rank_dims_signature, dims_signature);
}

TfLiteStatus Interpreter::SetExecutionPlan(size_t subgraph_index, const std::vector<int>& new_plan) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->SetExecutionPlan(new_plan);
}

void Interpreter::SetXNNPACKNumThreads(int num_threads) {
  if (num_threads < -1) {
    if (error_reporter_) {
      error_reporter_->Report(
          "num_threads should be >=0 or just -1 to let TFLite "
          "runtime set the value.");
    }
    return;
  }

#ifdef TFLITE_BUILD_WITH_XNNPACK_DELEGATE
  TfLiteDelegate* delegate = delegates(kTfLiteDelegateFlagsXNNPACK);
  if (delegate != nullptr) {
      TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
      // Modify -1 to 0 to match the XNNPACK runtime behavior 
      // to automatically set the value.
      if (num_threads == -1)
        num_threads = 0;
      options.num_threads = num_threads;
      TfLiteXNNPackDelegateUpdate(delegate, &options);
  }
#endif
}

void Interpreter::SetAllowFp16PrecisionForFp32(bool allow) {
  for (auto& subgraph : subgraphs_) {
    subgraph->context()->allow_fp32_relax_to_fp16 = allow;
  }
}

bool Interpreter::GetAllowFp16PrecisionForFp32() const {
  // TODO: #7, #77
  // Which context should we use here?
  // possibly move to external cpu contexts
  if (subgraphs_size()) {
    return subgraphs_[0]->context_.allow_fp32_relax_to_fp16;
  } else {
    return false;
  }
}

// TODO(b/121264966): Subgraphs added after cancellation is set will not get the
// cancellation function added to their context.
void Interpreter::SetCancellationFunction(void* data,
                                          bool (*check_cancelled_func)(void*)) {
  for (auto& subgraph : subgraphs_) {
    subgraph->SetCancellationFunction(data, check_cancelled_func);
  }
}


TfLiteStatus Interpreter::EnsureTensorDataIsReadable(size_t subgraph_index,
                                                     size_t tensor_index) {
  TF_LITE_ENSURE_SUBGRAPH_INDEX(subgraph_index);
  return subgraph(subgraph_index)->EnsureTensorDataIsReadable(tensor_index);
}

bool Interpreter::IsCancelled(size_t subgraph_index) {
  if (subgraph(subgraph_index)) {
    return subgraph(subgraph_index)->IsCancelled();
  } else {
    return false;
  }
}

bool Interpreter::HasDelegates(size_t subgraph_index) {
  assert(subgraph_index < subgraphs_.size());
  return subgraph(subgraph_index)->HasDelegates();
}

TfLiteStatus Interpreter::SetBufferHandle(size_t subgraph_index,
                                          size_t tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  // Use error_reporter_ to report error since context is not safe yet.
  TF_LITE_ENSURE(error_reporter_, subgraph_index < subgraphs_size());
  TF_LITE_ENSURE(error_reporter_, tensor_index < tensors_size(subgraph_index));
  std::vector<TfLiteTensor>& tensors = subgraph(subgraph_index)->tensors();
  TfLiteContext* context = &subgraph(subgraph_index)->context_;
  TfLiteTensor* tensor = &tensors[tensor_index];

  TF_LITE_ENSURE(context,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE(context, tensor->delegate->FreeBufferHandle != nullptr);
    tensor->delegate->FreeBufferHandle(context, tensor->delegate,
                                       &tensor->buffer_handle);
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(size_t subgraph_index,
                                          size_t tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
  TF_LITE_ENSURE(error_reporter_, subgraph_index < subgraphs_size());
  TF_LITE_ENSURE(error_reporter_, tensor_index < tensors_size(subgraph_index));
  std::vector<TfLiteTensor>& tensors = subgraph(subgraph_index)->tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

void Interpreter::UpdateExpectedLatency(
    const int subgraph_idx, int64_t latency) {
  int64_t prev_latency = moving_averaged_latencies_[subgraph_idx];
  moving_averaged_latencies_[subgraph_idx] =
      profile_smoothing_factor_ * latency +
      (1 - profile_smoothing_factor_) * prev_latency;
}

int64_t Interpreter::GetExpectedLatency(const int subgraph_idx) {
  auto it = moving_averaged_latencies_.find(subgraph_idx);
  if (it != moving_averaged_latencies_.end()) {
    return it->second;
  } else {
    return -1;
  }
}

int64_t Interpreter::GetProfiledLatency(SubgraphKey& key) {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second;
  } else {
    return -1;
  }
}

void Interpreter::Profile(int model_id) {
  tflite::Profiler* previous_profiler = GetProfiler();
  // Assign temporal time profiler for profiling.
  tflite::profiling::TimeProfiler timer;
  // Only update subgraph profilers to not care ownership of the profiler
  SetSubgraphProfiler(&timer);

  if (profile_online_) {
    ProfileOnline(model_id, timer);
  } else {
    ProfileOffline(model_id, timer);
  }

  SetSubgraphProfiler(previous_profiler);
  SetSLOBasedOnProfile();
}

void Interpreter::ProfileOnline(int model_id,
                                tflite::profiling::TimeProfiler& timer) {
  for (int worker_id = 0; worker_id < workers_.size(); worker_id++) {
    Worker* worker = workers_[worker_id].get();
    const char* device_name = TfLiteDeviceGetName(worker->GetDeviceFlag());

    // Get subgraphs for target model & worker
    std::vector<int> worker_subgraph_indices;
    for (int sub_idx = 0; sub_idx < subgraphs_size(); sub_idx++) {
      SubgraphKey& key = subgraphs_[sub_idx]->GetKey();
      if (key.model_id == model_id && key.worker_id == worker_id) {
        worker_subgraph_indices.push_back(sub_idx);
      }
    }
    if (worker_subgraph_indices.size() == 0) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "No subgraph for model %d and worker %d.", model_id,
                           worker_id);
      continue;
    }

    // Pause worker for profiling
    // Must resume before continue
    worker->Pause();
    int job_id = worker->GetCurrentJobId();
    if (job_id != -1) {
      planner_->Wait({job_id});
    }

    // Health check for subgraphs
    std::thread health_check_thread([&]() {
      bool all_healthy = true;
      SetProfileEnvironment(worker);
      for (const int& sub_idx : worker_subgraph_indices) {
        Subgraph* subgraph = subgraphs_[sub_idx].get();
        const SubgraphKey& key = subgraph->GetKey();
        if (subgraph->Invoke() != kTfLiteOk) {
          all_healthy = false;
          subgraph->SetHealth(false);
          moving_averaged_latencies_[sub_idx] = INT_MAX;
          profile_database_[key] = INT_MAX;

          TF_LITE_REPORT_ERROR(
              error_reporter_,
              "Subgraph %d execution failed for model %d and worker %d.",
              sub_idx, model_id, worker_id);
        }
      }
      if (all_healthy) {
        TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
                   "All subgraphs are executable for model %d worker %d.",
                   model_id, worker_id);
      }
    });
    health_check_thread.join();

    // Get largest subgraph
    int max_num_ops = -1;
    int max_subgraph_idx = -1;
    for (const int& sub_idx : worker_subgraph_indices) {
      const Subgraph* subgraph = subgraphs_[sub_idx].get();
      const SubgraphKey& key = subgraphs_[sub_idx]->GetKey();
      const int& num_ops = subgraph->op_indices().size();
      if (subgraph->GetHealth() && num_ops > max_num_ops) {
        max_num_ops = num_ops;
        max_subgraph_idx = sub_idx;
      }
    }
    if (max_subgraph_idx == -1) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "No executable subgraphs for model %d and worker %d.",
                           model_id, worker_id);
      worker->Resume();
      continue;
    }

    // Profile largest subgraph
    Subgraph* max_subgraph = subgraphs_[max_subgraph_idx].get();
    int64_t max_latency = ProfileSubgraph(max_subgraph, timer);
    if (max_latency < 0) {
      max_subgraph->SetHealth(false);
      moving_averaged_latencies_[max_subgraph_idx] = INT_MAX;
      profile_database_[max_subgraph->GetKey()] = INT_MAX;

      std::string msg = max_latency == -1 ? "Largest subgraph profile failed"
                                          : "Largest subgraph latency < 0";
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "%s for subgraph %d ,model %d and worker %d.",
                           msg.c_str(), max_subgraph_idx, model_id, worker_id);
      worker->Resume();
      continue;
    }

    const SubgraphKey& key = max_subgraph->GetKey();
    TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
               "Largest Subgraph Profiling result\n model=%d avg=%d us "
               "worker=%d device=%s start=%s end=%s.",
               model_id, max_latency, worker_id, device_name,
               key.GetInputOpsString().c_str(),
               key.GetOutputOpsString().c_str());
    LOGI("Largest Subgraph Profiling result\n model=%d avg=%d us worker=%d device=%s", model_id, max_latency, worker_id, device_name);

    // Resume worker
    worker->Resume();

    // Estimate latency with largest subgraph latency
    const Subgraph* primary_subgraph =
        subgraph(GetSubgraphIdx(model_id, kTfLiteCPU));
    for (const int& sub_idx : worker_subgraph_indices) {
      Subgraph* subgraph = subgraphs_[sub_idx].get();
      const SubgraphKey& key = subgraphs_[sub_idx]->GetKey();
      if (subgraph->GetHealth()) {
        const int64_t latency = EstimateLatency(
            subgraph, max_subgraph, primary_subgraph, max_latency,
            profile_copy_computation_ratio_[worker_id]);

        moving_averaged_latencies_[sub_idx] = latency;
        profile_database_[key] = latency;
        planner_->GetModelManager()->ProfileLatency(subgraph, latency);

        TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
                   "Estimated Latency\n model=%d avg=%d us worker=%d device=%s "
                   "start=%s end=%s.",
                   key.model_id, latency, key.worker_id, device_name,
                   key.GetInputOpsString().c_str(),
                   key.GetOutputOpsString().c_str());
        LOGI("Estimated Latency\n model=%d avg=%d us worker=%d device=%s",key.model_id, latency, key.worker_id, device_name);
      }
    }
  }
}

int64_t Interpreter::EstimateLatency(const Subgraph* target_subgraph,
                                     const Subgraph* max_subgraph,
                                     const Subgraph* primary_subgraph,
                                     int64_t max_latency,
                                     int64_t copy_computation_ratio) {
  int64_t target_flops = EstimateFLOPS(target_subgraph, primary_subgraph);
  int64_t target_size = EstimateInputOutputSize(target_subgraph);

  int64_t max_flops = EstimateFLOPS(max_subgraph, primary_subgraph);
  int64_t max_size = EstimateInputOutputSize(max_subgraph);

  int64_t estimated_latency =
      max_latency *
      (target_flops + target_size * copy_computation_ratio) /
      (max_flops + max_size * copy_computation_ratio);
  if (estimated_latency == 0) {
    return 1;
  } else {
    return estimated_latency;
  }
}

int64_t Interpreter::EstimateFLOPS(const Subgraph* subgraph,
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

int64_t Interpreter::EstimateInputOutputSize(const Subgraph* subgraph) {
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

void Interpreter::ProfileOffline(int model_id,
                                 tflite::profiling::TimeProfiler& timer) {
  for (int sub_idx = 0; sub_idx < subgraphs_size(); sub_idx++) {
    Subgraph* subgraph = subgraphs_[sub_idx].get();
    SubgraphKey& key = subgraph->GetKey();
    const char* device_name =
        TfLiteDeviceGetName(GetWorkerDeviceFlag(key.worker_id));

    if (key.model_id != model_id) {
      continue;
    }

    auto it = profile_database_.find(key);
    if (it != profile_database_.end()) {
      // TODO: Consider affinity of worker thread
      // if an entry for this SubgraphKey exists in the profiled data,
      // then reuse it to reduce initialization time
      int64_t profiled_latency = it->second;
      moving_averaged_latencies_[sub_idx] = profiled_latency;
      planner_->GetModelManager()->ProfileLatency(subgraph, profiled_latency);

      TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
                 "Reusing profiled result\n model=%d avg=%d us worker=%d "
                 "device=%s start=%s end=%s.",
                 key.model_id, profiled_latency, key.worker_id, device_name,
                 key.GetInputOpsString().c_str(),
                 key.GetOutputOpsString().c_str());
      LOGI("Reusing profiled result\n model=%d avg=%d us worker=%d device=%s",key.model_id, profiled_latency, key.worker_id, device_name);
    } else {
      int64_t latency = ProfileSubgraph(subgraph, timer);
      if (latency < 0) {
        subgraph->SetHealth(false);
        moving_averaged_latencies_[sub_idx] = INT_MAX;
        profile_database_[key] = INT_MAX;

        std::string msg =
            latency == -1 ? "Latency profile failed" : "Profiled latency < 0";
        TF_LITE_REPORT_ERROR(error_reporter_,
                            "%s for subgraph %d ,model %d and worker %d",
                            msg.c_str(), sub_idx, model_id, key.worker_id);
        continue;
      }

      moving_averaged_latencies_[sub_idx] = latency;
      profile_database_[key] = latency;
      planner_->GetModelManager()->ProfileLatency(subgraph, latency);

      TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
                 "Profiling result\n model=%d avg=%d us worker=%d "
                 "device=%s start=%s end=%s.",
                 key.model_id, latency, key.worker_id, device_name,
                 key.GetInputOpsString().c_str(),
                 key.GetOutputOpsString().c_str());
      LOGI("Profiling result\n model=%d avg=%d us worker=%d device=%s",key.model_id, latency, key.worker_id, device_name);
    }
  }
}

int64_t Interpreter::ProfileSubgraph(Subgraph* subgraph,
                                     tflite::profiling::TimeProfiler& timer) {
  int64_t latency = -1;

  std::thread t([&]() {
    SubgraphKey& subgraph_key = subgraph->GetKey();
    Worker* worker = workers_[subgraph_key.worker_id].get();
    SetProfileEnvironment(worker);
    for (int i = 0; i < profile_num_warmups_; i++) {
      if (subgraph->Invoke() != kTfLiteOk) {
        return;
      }
    }
    timer.ClearRecords();
    for (int i = 0; i < profile_num_runs_; i++) {
      if (subgraph->Invoke() != kTfLiteOk) {
        return;
      }
    }
    latency = timer.GetAverageElapsedTime<std::chrono::microseconds>();
  });
  t.join();

  return latency;
}

void Interpreter::SetProfileEnvironment(Worker* worker) {
  auto cpu_set = worker->GetWorkerThreadAffinity();
  SetCPUThreadAffinity(cpu_set);
  if (worker->GetDeviceFlag() == kTfLiteCPU) {
    auto internal_backend = GetCpuBackendContext()->internal_backend_context();
    // Update internal cpu backend (ruy)
    internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set);
    internal_backend->SetMaxNumThreads(worker->GetNumThreads());
  }
}

void Interpreter::SetProfiler(Profiler* profiler) {
  // Release resources occupied by owned_profiler_ which is replaced by
  // caller-owned profiler.
  owned_profiler_.reset(nullptr);
  installed_profiler_ = profiler;
  SetSubgraphProfiler(installed_profiler_);
}

void Interpreter::SetProfiler(std::unique_ptr<Profiler> profiler) {
  owned_profiler_ = std::move(profiler);
  installed_profiler_ = owned_profiler_.get();
  SetSubgraphProfiler(installed_profiler_);
}

void Interpreter::SetSubgraphProfiler(Profiler* profiler) {
  for (size_t subgraph_index = 0; subgraph_index < subgraphs_.size();
       ++subgraph_index) {
    subgraphs_[subgraph_index]->SetProfiler(profiler,
                                            subgraph_index);
  }
}

Profiler* Interpreter::GetProfiler() {
  if (installed_profiler_)
    return installed_profiler_;
  else if (owned_profiler_)
    return owned_profiler_.get();
  else
    return nullptr;
}

bool Interpreter::NeedProfile() {
  if (planner_)
    return planner_->NeedProfile();
  else
    return false;
}

TfLiteStatus Interpreter::ApplyBestDeviceDelegate(Subgraph* subgraph,
                                              TfLiteDeviceFlags device, 
                                              const std::set<TfLiteType>& tensor_types) {
  TfLiteDelegate* targetDelegate = nullptr;

  switch (device) {
    case kTfLiteCPU:
    case kTfLiteCLOUD:
      // TODO #23: XNNPACK seems inefficient than default CPU
      if (targetDelegate == nullptr)
        // Only valid case to return Ok with nullptr
        return kTfLiteOk;
      break;
    
    case kTfLiteGPU:
      targetDelegate = delegates(kTfLiteDelegateFlagsGPU);
      break;

    case kTfLiteDSP:
      if (tensor_types.find(kTfLiteInt8) != tensor_types.end() ||
          tensor_types.find(kTfLiteUInt8) != tensor_types.end())
        targetDelegate = delegates(kTfLiteDelegateFlagsNNAPIDSP);
      break;
      
    // TODO # 30
    // Add NPU / TPU / hta
    case kTfLiteNPU:
        targetDelegate = delegates(kTfLiteDelegateFlagsNNAPINPU);
      break;
    
    default:
      break;
  }

  if (targetDelegate != nullptr) {
    return subgraph->ModifyGraphWithDelegate(targetDelegate);
  } else {
    return kTfLiteError;
  }
}

void Interpreter::DeleteKey(SubgraphKey subgraph_key) {
  auto it = subgraph_idx_map_.find(subgraph_key);
  if (it != subgraph_idx_map_.end()) {
    subgraph_idx_map_.erase(it);
  }
}

int Interpreter::GetSubgraphIdx(SubgraphKey subgraph_key) {
  auto it = subgraph_idx_map_.find(subgraph_key);
  if (it != subgraph_idx_map_.end()) {
    return it->second;
  } else {
    return -1;
  }
}

TfLiteDeviceFlags Interpreter::GetWorkerDeviceFlag(int worker_id) {
  const Worker* worker = GetWorker(worker_id);
  if (worker) {
    return worker->GetDeviceFlag();
  } else {
    return kTfLiteNumDevices;
  }
}

int Interpreter::GetRepresentativeWorkerId(TfLiteDeviceFlags device_flag) {
  for(int worker_id = 0; worker_id < workers_.size(); worker_id++) {
    if (workers_[worker_id].get()->GetDeviceFlag() == device_flag) {
      return worker_id;
    }
  }
  return -1;
}

Worker* Interpreter::GetWorker(int worker_id) {
  if (worker_id >= 0 && worker_id < workers_.size()) {
    return workers_[worker_id].get();
  } else {
    return nullptr;
  }
}

std::set<int> Interpreter::GetSubgraphIdx(int model_id,
                                          int worker_id,
                                          int start_idx) {
  std::set<int> indices;
  for (auto& subgraph_key_id : subgraph_idx_map_) {
    const SubgraphKey& key = subgraph_key_id.first;
    int subgraph_index = subgraph_key_id.second;

    if (key.model_id == model_id && key.worker_id == worker_id
        && key.input_ops.find(start_idx) != key.input_ops.end()) {
      indices.insert(subgraph_index);
    }
  }

  return indices;
}

std::vector<int> Interpreter::GetSubgraphIndices(int model_id) {
  std::vector<int> indices;
  for (auto& subgraph_key_id : subgraph_idx_map_) {
    const SubgraphKey& key = subgraph_key_id.first;
    int subgraph_index = subgraph_key_id.second;

    if (key.model_id == model_id) {
      indices.push_back(subgraph_index);
    }
  }
  return indices;
}

int Interpreter::GetSubgraphIdx(int model_id, int worker_id) {
  // start_idx and end_idx weren't specified, so we assume that the caller
  // intended to retrieve the whole model
  ModelSpec& spec = model_specs_[model_id];
  for (int i = 0; i < subgraphs_size(); i++) {
    Subgraph* current_subgraph = subgraph(i);
    if (current_subgraph->key_.model_id == model_id &&
        current_subgraph->key_.worker_id == worker_id &&
        current_subgraph->prev_subgraphs_.size() == 0 &&
        current_subgraph->next_subgraphs_.size() == 0) {
      return i;
    }
  }
  return -1;
}

int Interpreter::GetSubgraphIdx(int model_id, TfLiteDeviceFlags device_flag) {
  return GetSubgraphIdx(model_id, GetRepresentativeWorkerId(device_flag));
}

std::set<int> Interpreter::models() const {
  std::set<int> models;
  for (auto& key : subgraph_idx_map_) {
    models.insert(key.first.model_id);
  }
  return models;
}

void Interpreter::SetModelConfigAndFillProfile(int model_id,
                                               ModelConfig& model_config) {
  SetModelConfig(model_id, model_config);

  // Set (model_id, start_unit_idx, end_unit_idx) -> subgraph idx map.
  for (int i = 0; i < subgraphs_.size(); ++i) {
    auto& subgraph_key = subgraphs_[i]->GetKey();
    if (subgraph_key.model_id != model_id) {
      continue;
    }
    int start_unit_idx = *subgraph_key.unit_indices.begin();
    int end_unit_idx = *subgraph_key.unit_indices.rbegin();

    unit_subgraphs_to_subgraph_indices_[model_id][start_unit_idx][end_unit_idx].push_back(i);
    TFLITE_LOG_INTERNAL(TFLITE_LOG_INFO,
               "Set unit subgraphs: model_id %d, start idx %d, end idx %d, "
               "subgraph idx %d",
               model_id, start_unit_idx, end_unit_idx, i);
  }

  std::string& model_fname = model_config.model_fname;
  auto model_profile =
      profiling::util::ExtractModelProfile(profile_database_json_,
                                           model_fname, model_id);

  // merge `profile_database_` with `model_profile`
  profile_database_.insert(model_profile.begin(), model_profile.end());
}

std::vector<std::pair<TfLiteDeviceFlags,std::set<int>>>
Interpreter::MakeSubgraphsForFallbackOps(const int model_id,
                                         const TfLiteDeviceFlags device_flag) {
  std::vector<std::pair<TfLiteDeviceFlags,std::set<int>>> subgraph_indices;
  const int num_ops = model_specs_[model_id].num_ops;
  const std::set<int>& unsupported_ops =
      model_specs_[model_id].unsupported_ops[device_flag];

  if (!planner_->NeedFallbackSubgraphs()) {
    return {{device_flag, {}}};
  }

  // TODO: Context-independent code / move to interpreter builder
  Subgraph* primary_subgraph = subgraph(GetSubgraphIdx(model_id, kTfLiteCPU));

  std::set<int> resolved_tensors;
  std::set<int> remaining_ops;
  // The basic idea is to partition this model into several disjoint subgraphs. 
  // Each subgraph is not necessarily a connected graph, and no two graphs
  // have any common ops. A subgraph is either a fallback subgraph or a
  // non-fallback one, but (obviously) never both.
  //
  //   Subgraph1  Sbg2     Sbg3
  // |--Non-fb--|--fb--|--Non-fb-|
  //
  //       Op2 --- Op3 -- Op4
  //     /                   \
  // Op1 - Op5 --- Op6 -- Op7 - Op8
  //
  // We start from the foremost op(s) and gradually "expand" our territory of
  // ops until we have the largest subgraph possible, without going over the
  // boundary of fallback/non-fallback. After that, we remove the ops of that
  // largest subgraph and start over with the remaining ops. This process is
  // repeated until all ops have been removed.

  // To make this work, we first need to keep track of the "front line" of ops.
  // This front line, together with the fallback/non-fb status of the op,
  // is used to determine whether or not we include an op in the current
  // subgraph.
  // The front line is denoted with the set of "resolved" tensors -- a tensor
  // is considered resolved if that tensor can be computed using external
  // inputs + previously resolved tensors. In case all input tensors of an 
  // op are resolved ones, that op is regarded to be at the front line of ops
  // and thus can be put into the current subgraph (+ the fb/non-fb status
  // must match too).
  for (int input_index : primary_subgraph->inputs()) {
    resolved_tensors.insert(input_index);
  }

  for (int i = 0; i < num_ops; i++) {
    remaining_ops.insert(i);
  }

  // convenience function for determining if a tensor has been resolved
  auto is_resolved = [&](int op_index) {
    auto op_inputs =
        primary_subgraph->node_and_registration(op_index)->first.inputs;
    for (int i = 0; i < op_inputs->size; i++) {
      if (primary_subgraph->tensor(op_inputs->data[i])->allocation_type == kTfLiteMmapRo) {
        // parameter tensors are always available,
        // so they always count as "resolved" tensors
        continue;
      }
      if (resolved_tensors.find(op_inputs->data[i]) == resolved_tensors.end()) {
        return false;
      }
    }
    return true;
  };

  bool is_fallback = false;
  while (remaining_ops.size() > 0) {
    std::set<int> operator_set;
    bool found = true;
    // Switch between device and fallback 
    TfLiteDeviceFlags current_device = 
        is_fallback ?
        kTfLiteCPU : device_flag;

    // Get all op that has resolvable dependency to specific device
    while (found) {
      found = false;
      for (auto current_op = remaining_ops.begin();
           current_op != remaining_ops.end();) {
        int current_index = *current_op;
        bool is_op_unsupported =
            unsupported_ops.find(current_index) != unsupported_ops.end();
        if (!is_fallback == is_op_unsupported) {
          // either 1) this is a fallback op but we're making a non-fb subgraph,
          // or 2) this is a non-fb op but we're making a fb subgraph,
          // so we skip it
          current_op++;
          continue;
        }

        // Dependency check
        if (!is_resolved(current_index)) {
          current_op++;
          continue;
        }

        found = true;
        operator_set.insert(current_index);

        auto op_outputs =
            primary_subgraph->node_and_registration(current_index)->first.outputs;
        
        // Update dependency to include output tensors of this new op.
        // This has the effect of expanding the "front line" of ops.
        for (int i = 0; i < op_outputs->size; i++) {
          resolved_tensors.insert(op_outputs->data[i]);
        }

        current_op = remaining_ops.erase(current_op);
      }
    }  

    if (operator_set.size()) {
      subgraph_indices.push_back({current_device, operator_set});
    }

    is_fallback = !is_fallback;
  }

  return subgraph_indices;
}

TfLiteStatus Interpreter::GetUnitSubgraphs(
    const int model_id, std::set<std::pair<int, DeviceOpIndices>>& subgraph_indices,
    bool need_fallback_subgraph) {
  if (!need_fallback_subgraph) {
    for (auto& worker : workers_) {
      TfLiteDeviceFlags device_flag = worker->GetDeviceFlag();
      subgraph_indices.insert({0, {device_flag, {}}});
    }
    PrepareUnitSubgraphScheduling(model_id, 1);
    return kTfLiteOk;
  }

  // Prepare variables to use
  const int num_ops = model_specs_[model_id].num_ops;
  Subgraph* primary_subgraph = subgraph(GetSubgraphIdx(model_id, kTfLiteCPU));

  // BitMask to check device support or not
  using BitMask = uint32_t;
  if (kTfLiteNumDevices > 8 * sizeof(BitMask)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "kTfLiteNumDevices is larger than BitMask %d",
                         kTfLiteNumDevices);
  }

  std::map<TfLiteDeviceFlags, std::set<int>> op_sets_to_ignore;
  // register subgraphs for all devices
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);
    std::vector<DeviceOpIndices> device_op_sets =
      MakeSubgraphsForFallbackOps(model_id, device_flag);
    for (auto device_and_ops : device_op_sets) {
      auto device = device_and_ops.first;
      auto& ops = device_and_ops.second;
      if (device == kTfLiteCPU || device == kTfLiteCLOUD) {
        continue;
      }
      if (ops.size() < minimum_subgraph_size_) {
        for (auto op : ops) {
          op_sets_to_ignore[device].insert(op);
        }
      }
    }
  }

  // Build op_support_table
  std::vector<BitMask> op_support_table(num_ops, 0U);
  const std::map<TfLiteDeviceFlags, std::set<int>>& unsupported_ops =
      model_specs_[model_id].unsupported_ops;
  for (int op_index = 0; op_index < num_ops; op_index++) {
    for (int device_id = 0; device_id < kTfLiteNumDevices; device_id++) {
      TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_id);
      if (device_flag == kTfLiteCPU || device_flag == kTfLiteCLOUD) {
        op_support_table[op_index] |= 1 << device_id;
        continue;
      }
      if (unsupported_ops.find(device_flag) == unsupported_ops.end() ||
          unsupported_ops.at(device_flag).find(op_index) == unsupported_ops.at(device_flag).end()) {
        if (op_sets_to_ignore[device_flag].find(op_index) == op_sets_to_ignore[device_flag].end()) {
          op_support_table[op_index] |= 1 << device_id;
        }
      }
    }
  }

  // TODO: Add description
  // Add unit Subgraphs.
  // Op indices in single unit subgraph have same support devices.
  std::vector<bool> is_resolved_tensor(primary_subgraph->tensors_size(), false);
  std::set<int> remaining_ops;

  for (int input_index : primary_subgraph->inputs()) {
    is_resolved_tensor[input_index] = true;
  }

  for (int i = 0; i < num_ops; i++) {
    remaining_ops.insert(i);
  }

  // convenience function for determining if op inputs are resolved
  auto is_resolved_op = [&primary_subgraph, &is_resolved_tensor](int op_index) {
    auto op_inputs =
        primary_subgraph->node_and_registration(op_index)->first.inputs;
    for (int i = 0; i < op_inputs->size; i++) {
      if (primary_subgraph->tensor(op_inputs->data[i])->allocation_type == kTfLiteMmapRo) {
        // parameter tensors are always available,
        // so they always count as "resolved" tensors
        continue;
      }
      if (!is_resolved_tensor[op_inputs->data[i]]) {
        return false;
      }
    }
    return true;
  };

  int subgraph_local_idx = 0;
  while (true) {
    std::set<int> unit_subgraph_ops;
    BitMask support_devices = 0;

    // Find single unit subgraph ops
    while (true) {
      // Find addable ops
      // 1. resolved
      // 2. same support devices
      std::vector<int> to_add;
      for (int op_index : remaining_ops) {
        // Check the op is resolved
        if (!is_resolved_op(op_index)) {
          continue;
        }
        // Check the op have same support devices
        if (support_devices != 0 &&
            support_devices != op_support_table[op_index]) {
          continue;
        }
        // Set support devices using first op
        if (support_devices == 0) {
          support_devices = op_support_table[op_index];
        }
        to_add.push_back(op_index);
      }
      // If there is no more ops to add, stop
      if (to_add.empty()) break;

      // Add ops which are resolved and have same support devices
      unit_subgraph_ops.insert(to_add.begin(), to_add.end());

      // Delete resolved ops and add resolved tensors
      for (int op_index : to_add) {
        remaining_ops.erase(remaining_ops.find(op_index));
        auto op_outputs =
            primary_subgraph->node_and_registration(op_index)->first.outputs;
        for (int i = 0; i < op_outputs->size; i++) {
          is_resolved_tensor[op_outputs->data[i]] = true;
        }
      }
    }
    if (unit_subgraph_ops.empty()) break;
    for (int device_id = 0; device_id < kTfLiteNumDevices; device_id++) {
      if (support_devices & (1 << device_id)) {
        TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_id);
        subgraph_indices.insert({subgraph_local_idx, {device_flag, unit_subgraph_ops}});
      }
    }
    subgraph_local_idx++;
  }
  if (!remaining_ops.empty()) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Not empty remaining ops");
    return kTfLiteError;
  }
  PrepareUnitSubgraphScheduling(model_id, subgraph_local_idx);

  return kTfLiteOk;
}

void Interpreter::InvestigateModelSpec(int model_id) {
  // get the subgraph index for this model
  int worker_id = GetRepresentativeWorkerId(kTfLiteCPU);
  int subgraph_index = GetSubgraphIdx(model_id, worker_id);
  Subgraph* primary_subgraph = subgraph(subgraph_index);

  // this creates an empty ModelSpec
  ModelSpec& model_spec = model_specs_[model_id];

  std::vector<int>& execution_plan = primary_subgraph->execution_plan();
  model_spec.num_ops = execution_plan.size();

  // allocate circular buffer for model IO
  std::vector<TfLiteTensor*> input_tensors;
  std::vector<TfLiteTensor*> output_tensors;

  for (int input_tensor : primary_subgraph->inputs()) {
    input_tensors.push_back(primary_subgraph->tensor(input_tensor));
  }

  for (int output_tensor : primary_subgraph->outputs()) {
    output_tensors.push_back(primary_subgraph->tensor(output_tensor));
  }

  model_input_buffer_.emplace(model_id, std::make_unique<TensorRingBuffer>(
                                            error_reporter_, input_tensors,
                                            primary_subgraph->inputs()));
  model_output_buffer_.emplace(model_id, std::make_unique<TensorRingBuffer>(
                                             error_reporter_, output_tensors,
                                             primary_subgraph->outputs()));

  // check input/output/intermediate tensors to fill in
  // model_spec.output_tensors and model_spec.tensor_types
  for (auto node_index : execution_plan) {
    const TfLiteNode& node =
        primary_subgraph->node_and_registration(node_index)->first;

    std::set<int> tensor_indices;
    for (int input_tensor : TfLiteIntArrayView(node.inputs)) {
      tensor_indices.insert(input_tensor);
    }

    for (int output_tensor : TfLiteIntArrayView(node.outputs)) {
      tensor_indices.insert(output_tensor);
      model_spec.node_output_tensors.insert(output_tensor);
    }

    for (auto i : tensor_indices) {
      const auto* tensor = primary_subgraph->tensor(i);
      model_spec.tensor_types.insert(tensor->type);
    }
  }

  std::copy(primary_subgraph->inputs().begin(),
            primary_subgraph->inputs().end(),
            std::inserter(model_spec.input_tensors,
                          model_spec.input_tensors.begin()));

  std::copy(primary_subgraph->outputs().begin(),
            primary_subgraph->outputs().end(),
            std::inserter(model_spec.output_tensors,
                          model_spec.output_tensors.begin()));

  // also check unsupported ops to fill in model_spec.unsupported_ops
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);

    if (device_flag == kTfLiteCPU || device_flag == kTfLiteCLOUD) {
      // no need to check supportability for CPU
      continue;
    }

    // try creating a delegate for this device
    // ops (`node` below) that weren't converted are the unsupported ops
    ApplyBestDeviceDelegate(primary_subgraph, device_flag,
                            model_spec.tensor_types);
    for (auto node_index : execution_plan) {
      const TfLiteNode& node =
          primary_subgraph->node_and_registration(node_index)->first;
      if (node.delegate == nullptr) {
        // this subgraph is always a 0~num_ops-1 CPU subgraph so
        // the node-->op mapping is basically the identity mapping
        model_spec.unsupported_ops[device_flag].insert(node_index);
      }
    }

    // revert changes
    primary_subgraph->RemoveAllDelegates();
  }
}

std::pair<std::vector<int>, int64_t> Interpreter::GetShortestLatencyWithUnitSubgraph(
    int model_id, int start_unit_idx,
    std::map<int, int64_t>& worker_waiting) {
  std::pair<std::vector<int>, int64_t> local_min = std::make_pair<std::vector<int>, int64_t>({}, -1);
  auto range = GetSubgraphIndices(model_id);
  std::pair<int, int64_t> target_subgraph = GetShortestSubgraphIndex(range, 0, worker_waiting);
  local_min.first.push_back(target_subgraph.first);
  local_min.second = target_subgraph.second;
  return local_min;
}

std::pair<int, int64_t> Interpreter::GetShortestLatency(
    int model_id, std::set<int> resolved_tensors, int64_t start_time,
    std::map<int, int64_t>& worker_waiting,
    int preceded_subgraph_index) {
  // lookup key for cache_, below
  std::pair<int, std::set<int>> cache_key = {model_id, resolved_tensors};

  // check if it is safe to lookup the cache:
  // are all waiting times < start_time ?
  bool wait_time_is_stale = true;
  for (auto& pair : worker_waiting) {
    auto wait_time = pair.second;
    if (wait_time > start_time) {
      wait_time_is_stale = false;
    }
  }

  if (wait_time_is_stale) {
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
      auto& pair = it->second;
      int subgraph_idx = pair.first;
      int64_t latency = pair.second;

      // the stored latency value assumes a start_time of 0,
      // so we need to add our own start_time to the stored value to get the
      // correct return value
      return {subgraph_idx, latency + start_time};
    }
  }

  std::vector<int> subgraph_indices =
      GetSubgraphCandidates(model_id, resolved_tensors, preceded_subgraph_index);
  std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>
      subgraph_map = GroupByStartEndIdx(subgraph_indices);

  std::pair<int, int64_t> min_subgraph = {-1, INT_MAX};
  for (auto iter = subgraph_map.begin(); iter != subgraph_map.end(); iter++) {
    // first, filter out the subgraphs that take longer than others with the
    // same start/end indices, since there's no reason to pick them
    std::pair<int, int64_t> target_subgraph =
        GetShortestSubgraphIndex(iter->second, start_time, worker_waiting);
    Subgraph* subgraph_ptr = subgraph(target_subgraph.first);
    SubgraphKey& key = subgraph_ptr->GetKey();

    std::set<int> next_resolved_tensors = resolved_tensors;

    // add current subgraph's output tensors to resolved_tensors 
    std::copy(
        subgraph_ptr->outputs().begin(), subgraph_ptr->outputs().end(),
        std::inserter(next_resolved_tensors, next_resolved_tensors.begin()));

    std::pair<int, int64_t> local_min;
    // all output tensors of the model is resolved
    if (std::includes(
            next_resolved_tensors.begin(), next_resolved_tensors.end(),
            GetModelSpec(model_id).output_tensors.begin(),
            GetModelSpec(model_id).output_tensors.end())) {
      local_min = target_subgraph;
    } else {
      // there's more ops left for this model, so we need to look further to
      // get the final latency
      local_min =
          GetShortestLatency(model_id, next_resolved_tensors, target_subgraph.second,
                             worker_waiting, target_subgraph.first);
    }

    // check if this subgraph is better than the best one
    if (local_min.second < min_subgraph.second) {
      // note the subgraph to return is the next immediate one (start_idx, XX),
      // but the latency to return is that of the final subgraph (XX, #ops)
      // hence, target_subgraph.first & local_min.second
      min_subgraph.first = target_subgraph.first;
      min_subgraph.second = local_min.second;
    }
  }

  if (wait_time_is_stale) {
    // if we've reached this point, then there shouldn't be an entry
    // for this key in the cache
    assert(cache_.find(cache_key) == cache_.end());

    // we are going to store the latency value for start_time == 0,
    // so do a sanity check for latency - start_time
    assert(min_subgraph.second >= start_time);

    cache_[cache_key] = {min_subgraph.first,
                         min_subgraph.second - start_time};
  }

  return min_subgraph;
}

int Interpreter::GetSubgraphIdxSatisfyingSLO(Job& job,
                                             std::map<int, int64_t>& worker_waiting,
                                             std::set<int>& idle_workers) {
  // TODO: support models with fallback ops.
  int target_subgraph_idx = -1;
  int model_id = job.model_id;
  auto num_unit_subgraphs = model_specs_[model_id].num_unit_subgraphs;
  auto& range = unit_subgraphs_to_subgraph_indices_[model_id][0][num_unit_subgraphs - 1];

  if (range.size() == 0) {
    return -1;
  }

  bool satisfy_slo = false;
  // NOTE: Consider changing to `max_expected_latency`,
  // to yield faster accelerators to following requests.
  int64_t min_expected_latency = -1;
  for (auto subgraph_index : range) {
    SubgraphKey& key = subgraph(subgraph_index)->GetKey();
    if (!subgraph(subgraph_index)->GetHealth()) {
      continue;
    }
    int64_t waiting_time = worker_waiting[key.worker_id];
    int64_t expected_execution_time = GetExpectedLatency(subgraph_index);
    int64_t current_time = profiling::time::NowMicros();
    int64_t expected_latency = expected_execution_time + waiting_time;

    if (current_time + expected_latency < job.enqueue_time + job.slo_us) {
      satisfy_slo = true;
      if (min_expected_latency == -1 || expected_latency < min_expected_latency) {
        if (idle_workers.find(key.worker_id) != idle_workers.end()) {
          min_expected_latency = expected_latency;
          target_subgraph_idx = subgraph_index;
        }
      }
    }
  }

  if (!satisfy_slo) {
    // If all the subgraphs cannot satisfy the slo,
    // then enqueue any subgraph.
    // `HandleSLOViolatedJob` will deal with the rest.
    target_subgraph_idx = range[0];
  }

  return target_subgraph_idx;
}

std::pair<std::vector<int>, int64_t>
Interpreter::GetSubgraphWithShortestLatency(Job& job,
                                            std::map<int, int64_t>& worker_waiting) {
  return GetShortestLatencyWithUnitSubgraph(job.model_id, job.start_unit_idx, worker_waiting);
}

std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>
Interpreter::GroupByStartEndIdx(
    std::vector<int> subgraph_indices) {
  std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>> ret;
  for (auto subgraph_index : subgraph_indices) {
    SubgraphKey& key = subgraph(subgraph_index)->GetKey();
    ret[{key.input_ops, key.output_ops}].push_back(
        subgraph_index);
  }
  return ret;
}

std::vector<int> Interpreter::GetSubgraphCandidates(
    int model_id, std::set<int> resolved_tensors, int preceded_subgraph_index) {
  std::vector<int> candidate_indices;
  // Start of the model execution
  if (preceded_subgraph_index == -1) {
    for (int i = 0; i < subgraphs_size(); ++i) {
      Subgraph* subgraph_ptr = subgraph(i);
      SubgraphKey& key = subgraph_ptr->GetKey();

      if (key.model_id == model_id && subgraph_ptr->IsStart()) {
        candidate_indices.push_back(i);
      }
    }
  } else {
    Subgraph* subgraph_ptr = subgraph(preceded_subgraph_index);
    auto key = subgraph_ptr->GetKey();
    for (Subgraph* next_subgraph : subgraph_ptr->GetNextSubgraphs()) {
      bool is_executable = true;
      // check whether all input tensor is resolved or not
      for (const int& input_tensor : next_subgraph->inputs()) {
        if (resolved_tensors.find(input_tensor) == resolved_tensors.end()) {
          is_executable = false;
          break;
        }
      }
      
      // TODO: Update with subgraph dependency generation logic
      // check whether any output tensor is resolved or not
      for (const int& output_tensor: next_subgraph->outputs()) {
        if (resolved_tensors.find(output_tensor) != resolved_tensors.end()) {
          is_executable = false;
          break;
        }
      }

      if (is_executable) {
        candidate_indices.push_back(GetSubgraphIdx(next_subgraph->GetKey()));
      }
    }
  }
  return candidate_indices;
}

std::pair<int, int64_t>
Interpreter::GetShortestSubgraphIndex(
    std::vector<int>& subgraph_indices, int64_t start_time,
    std::map<int, int64_t>& worker_waiting) {
  int64_t min_latency = INT_MAX;
  int min_idx = -1;

  for (auto subgraph_index : subgraph_indices) {
    SubgraphKey& key = subgraph(subgraph_index)->GetKey();
    if (!subgraph(subgraph_index)->GetHealth()) {
      continue;
    }

    int64_t waiting_time = worker_waiting[key.worker_id];
    int64_t expected_latency = GetExpectedLatency(subgraph_index);
    int64_t total = expected_latency + std::max(waiting_time, start_time);

    if (min_latency > total) {
      min_latency = total;
      min_idx = subgraph_index;
    }
  }
  return {min_idx, min_latency};
}

void Interpreter::SetSLOBasedOnProfile() {
  for (auto& m : model_configs_) {
    int model_id = m.first;
    ModelConfig& config = m.second;

    if (config.slo_us > 0) {
      // slo has already been set by the model json config file
      continue;
    }

    if (config.slo_scale <= 0) {
      // this model doesn't have an slo
      continue;
    }

    int64_t worst_latency = GetWorstDeviceProfileResult(model_id);
    config.slo_us = worst_latency * config.slo_scale;
  }
}

int64_t Interpreter::GetWorstDeviceProfileResult(int model_id) {
  int64_t worst_latency = 0;
  for (int i = 0; i < subgraphs_size(); ++i) {
    SubgraphKey& subgraph_key = subgraphs_[i]->GetKey();
    if (subgraph_key.model_id != model_id) {
      continue;
    }

    auto it = moving_averaged_latencies_.find(i);
    if (it != moving_averaged_latencies_.end()) {
      int64_t latency = it->second;
      if (worst_latency < latency) {
        worst_latency = latency;
      }
    }
  }

  if (worst_latency == 0) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Model %d has no profile results, but "
                         "GetWorstDeviceProfileResult was called",
                         model_id);
  }

  return worst_latency;
}

void Interpreter::PrepareUnitSubgraphScheduling(int model_id, int num_units) {
  auto& model_spec = model_specs_[model_id];
  model_spec.num_unit_subgraphs = num_units;
  model_spec.latency_memo.resize(num_units);

  Subgraph* primary_subgraph = subgraph(GetSubgraphIdx(model_id, kTfLiteCPU));
  for (int i = 0; i < num_units; ++i) {
    primary_subgraph->GetKey().unit_indices.insert(i);
  }
}

}  // namespace impl

}  // namespace tflite
