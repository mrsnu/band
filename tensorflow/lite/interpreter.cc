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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/context_util.h"
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
#include "tensorflow/lite/tools/logging.h"

// TODO(b/139446230): Move to portable platform header.
#if defined(__ANDROID__)
#include "tensorflow/lite/nnapi/nnapi_util.h"
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
  TFLITE_LOG(INFO) << "Initialized TensorFlow Lite runtime.";

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
  planner_.reset(new Planner(this));
  if (planner_->Init(runtime_config.planner_config) != kTfLiteOk) {
    error_reporter_->Report("Planner::Init() failed.");
    exit(-1);
  }
  // Initialize configurations.
  if (Init(runtime_config.interpreter_config) != kTfLiteOk) {
    error_reporter_->Report("Interpreter::Init() failed.");
    exit(-1);
  }

  std::set<TfLiteDeviceFlags> valid_devices = { kTfLiteCPU };
  if (planner_->NeedFallbackSubgraphs()) {
    valid_devices.insert(kTfLiteCPUFallback);
  }

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
  for (const char* device_name : string_device_names_list) {
    if (IsNNAPIDeviceUseful(device_name)) {
      TFLITE_LOG(INFO) << "Available NNAPI device name: " << device_name;
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
  for (const TfLiteDeviceFlags device_flag : valid_devices) {
    if (planner_->GetWorkerType() == kGlobalQueue) {
      workers_[device_flag] = std::make_unique<GlobalQueueWorker>(planner_, device_flag);
    } else {
      workers_[device_flag] = std::make_unique<DeviceQueueWorker>(planner_, device_flag);
    }
  }
  // Initialize workers.
  for (auto& worker : workers_) {
    if (worker.second->Init(runtime_config.worker_config) != kTfLiteOk) {
      error_reporter_->Report("Worker::Init() failed for worker : %s",
                               TfLiteDeviceGetName(worker.first));
      exit(-1);
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

  if (NeedProfile()) {
    profile_data_path_ = config.profile_data_path;
    profile_database_json_ = LoadJsonObjectFromFile(config.profile_data_path);
    // we cannot convert the model name strings to integer ids yet,
    // (profile_database_json_ --> profile_database_)
    // since we don't have anything in model_configs_ at the moment

    // Set how many runs are required to get the profile results.
    num_warmups_ = config.profile_config.num_warmups;
    num_runs_ = config.profile_config.num_runs;

    TFLITE_LOG(INFO) << "Set Profiling Configuration:"
                     << " warmup=" << num_warmups_
                     << " count=" << num_runs_;
  }

  const TfLiteCPUMaskFlags cpu_mask = 
      static_cast<TfLiteCPUMaskFlags>(config.cpu_masks);
  auto cpu_mask_set = TfLiteCPUMaskGetSet(cpu_mask);

  TFLITE_LOG(INFO) << "Set affinity to "
                   << TfLiteCPUMaskGetName(cpu_mask) << " cores";

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

std::vector<int> Interpreter::InvokeModelsAsync(std::vector<Tensors> inputs) {
  std::vector<Job> requests;
  std::vector<Tensors> duplicate_inputs;

  if (inputs.size() != model_configs_.size()) {
    error_reporter_->Report(
        "Invalid input size input.size() %d != model_configs_.size() %d",
        inputs.size(), model_configs_.size());
    return {};
  }

  for (auto& m : model_configs_) {
    int model_id = m.first;
    ModelConfig& model_config = m.second;
    Job request = Job(model_id);
    for (int k = 0; k < model_config.batch_size; ++k) {
      requests.push_back(request);
      duplicate_inputs.push_back(inputs[model_id]);
    }
  }

  return InvokeModelsAsync(requests, duplicate_inputs);
}

std::vector<int> Interpreter::InvokeModelsAsync(std::vector<Job> requests, 
                                                std::vector<Tensors> inputs) {
  if (requests.size() == 0) {
    return {};
  }

  for (auto& request: requests) {
    int model_id = request.model_id;
    ModelConfig& model_config = model_configs_[model_id];
    request.model_fname = model_config.model_fname;
    request.device_id = model_config.device;
    request.slo_us = model_config.slo_us;
  }

  std::vector<Job> valid_requests;
  std::vector<bool> valid_requests_masks(requests.size(), true);

  if (inputs.size() > 0) {
    assert(inputs.size() == requests.size());
    for (size_t i = 0; i < requests.size(); i++) {
      Job& request = requests[i];
      int input_handle = model_input_buffer_[request.model_id]->Alloc();
      if (model_input_buffer_[request.model_id]->PutTensorsToHandle(
              inputs[i], input_handle) == kTfLiteOk) {
        request.input_handle = input_handle;
        request.output_handle = model_output_buffer_[request.model_id]->Alloc();
        valid_requests.push_back(std::move(request));
      } else {
        valid_requests_masks[i] = false;
      }
    }
  } else {
    // we don't care about the inputs, just make input/output-less requests
    for (Job& request : requests) {
      valid_requests.push_back(std::move(request));
    }
  }

  std::vector<int> job_ids(requests.size(), -1);
  std::vector<int> valid_job_ids = planner_->EnqueueBatch(valid_requests);

  int valid_index = 0;
  for (size_t i = 0; i < requests.size(); i++) {
    if (valid_requests_masks[i]) {
      job_ids[i] = valid_job_ids[valid_index++];
    }
  }
  
  return job_ids;
}

void Interpreter::InvokeModelsSync(std::vector<Tensors> inputs,
                                   std::vector<Tensors> outputs) {
  std::vector<int> job_ids = InvokeModelsAsync(inputs);
  planner_->Wait(job_ids);
  
  if (inputs.size() == outputs.size()) {
    int accumulated_job_index = 0;
    int output_index = 0;
    for (auto& m : model_configs_) {
      GetOutputTensors(job_ids[accumulated_job_index], outputs[output_index++]);
      accumulated_job_index += m.second.batch_size;
    }
  }
}

void Interpreter::InvokeModelsSync(std::vector<Job> requests,
                                   std::vector<Tensors> inputs,
                                   std::vector<Tensors> outputs) {
  std::vector<int> job_ids = InvokeModelsAsync(requests, inputs);
  planner_->Wait(job_ids);
  for (size_t i = 0; i < job_ids.size(); i++) {
    GetOutputTensors(job_ids[i], outputs[i]);
  }
}

TfLiteStatus Interpreter::GetOutputTensors(int job_id, Tensors& outputs) const {
  Job job = planner_->GetFinishedJob(job_id);

  if (job.job_id == -1) {
    // Not finished yet.
    return kTfLiteOk;
  }

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    error_reporter_->Report("Invalid model_id : %d", job.model_id);
    return kTfLiteError;
  }

  return model_output_buffer_.at(job.model_id)
      ->GetTensorsFromHandle(outputs, job.output_handle);
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
    const SubgraphKey& key, int64_t latency) {
  int64_t prev_latency = moving_averaged_latencies_[key];
  moving_averaged_latencies_[key] =
      profile_smoothing_factor_ * latency +
      (1 - profile_smoothing_factor_) * prev_latency;
}

int64_t Interpreter::GetExpectedLatency(const SubgraphKey& key) {
  auto it = moving_averaged_latencies_.find(key);
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

  for (int i = 0; i < subgraphs_size(); ++i) {
    Subgraph* subgraph = subgraphs_[i].get();
    SubgraphKey& subgraph_key = subgraph->GetKey();

    if (subgraph_key.model_id != model_id) {
      continue;
    }

    auto it = profile_database_.find(subgraph_key);
    if (it != profile_database_.end()) {
      // if an entry for this SubgraphKey exists in the profiled data,
      // then reuse it to reduce initialization time
      int64_t profiled_latency = it->second;
      // TODO: Consider affinity of worker thread
      moving_averaged_latencies_[subgraph_key] = profiled_latency;

      TFLITE_LOG(INFO) << "Reusing profiled result\n"
                       << " model=" << subgraph_key.model_id
                       << " avg=" << profiled_latency << " us"
                       << " device="
                       << TfLiteDeviceGetName(subgraph_key.device_flag)
                       << " start="
                       << subgraph_key.GetInputOpsString()
                       << " end=" 
                       << subgraph_key.GetOutputOpsString() << ".";

    } else {
      std::thread t([&]() {
        auto cpu_set =
            workers_[subgraph_key.device_flag]->GetWorkerThreadAffinity();
        SetCPUThreadAffinity(cpu_set);
        if (subgraph_key.device_flag == kTfLiteCPU ||
            subgraph_key.device_flag == kTfLiteCPUFallback) {
          auto internal_backend =
              GetCpuBackendContext()->internal_backend_context();
          // Update internal cpu backend (ruy)
          internal_backend->SetCpuSet(std::this_thread::get_id(), cpu_set);
          internal_backend->SetMaxNumThreads(
              workers_[subgraph_key.device_flag]->GetNumThreads());
        }

        for (int i = 0; i < num_warmups_; i++) {
          subgraph->Invoke();
        }
        timer.ClearRecords();
        for (int i = 0; i < num_runs_; i++) {
          subgraph->Invoke();
        }

        int64_t latency = timer.GetAverageElapsedTime<std::chrono::microseconds>();

        moving_averaged_latencies_[subgraph_key] = latency;
        // record the profiled latency for subsequent benchmark runs
        profile_database_[subgraph_key] = latency;

        TFLITE_LOG(INFO) << "Profiling result\n"
                         << " model=" << subgraph_key.model_id
                         << " avg=" << latency << " us"
                         << " device=" << TfLiteDeviceGetName(subgraph_key.device_flag)
                         << " start="
                         << subgraph_key.GetInputOpsString()
                         << " end=" 
                         << subgraph_key.GetOutputOpsString() << ".";
      });
      t.join();
    }
  }

  SetSubgraphProfiler(previous_profiler);
  SetSLOBasedOnProfile();
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
    case kTfLiteCPUFallback:
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

Worker* Interpreter::GetWorker(int device_idx) {
  TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_idx);
  return GetWorker(device_flag);
}

Worker* Interpreter::GetWorker(TfLiteDeviceFlags device_flag) {
  auto it = workers_.find(device_flag);
  if (it != workers_.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

std::set<int> Interpreter::GetSubgraphIdx(int model_id,
                                          TfLiteDeviceFlags device_id,
                                          int start_idx) {
  std::set<int> indices;
  for (auto& subgraph_key_id : subgraph_idx_map_) {
    const SubgraphKey& key = subgraph_key_id.first;
    int subgraph_index = subgraph_key_id.second;

    if (key.model_id == model_id && key.device_flag == device_id
        && key.input_ops.find(start_idx) != key.input_ops.end()) {
      indices.insert(subgraph_index);
    }
  }

  return indices;
}

int Interpreter::GetSubgraphIdx(int model_id, int device_idx) {
  TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_idx);
  return GetSubgraphIdx(model_id, device_flag);
}

int Interpreter::GetSubgraphIdx(int model_id, TfLiteDeviceFlags device_id) {
  // start_idx and end_idx weren't specified, so we assume that the caller
  // intended to retrieve the whole model
  ModelSpec& spec = model_specs_[model_id];
  for (int i = 0; i < subgraphs_size(); i++) {
    Subgraph* current_subgraph = subgraph(i);
    if (current_subgraph->key_.model_id == model_id &&
        current_subgraph->key_.device_flag == device_id &&
        current_subgraph->prev_subgraphs_.size() == 0 &&
        current_subgraph->next_subgraphs_.size() == 0) {
      return i;
    }
  }
  return -1;
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
  std::string& model_fname = model_config.model_fname;

  auto model_profile =
      profiling::util::ExtractModelProfile(profile_database_json_,
                                           model_fname, model_id);

  // merge `profile_database_` with `model_profile`
  profile_database_.insert(model_profile.begin(), model_profile.end());
}

std::vector<DeviceOpIndices> Interpreter::MakeSubgraphsForFallbackOps(
    const int model_id, const TfLiteDeviceFlags device_flag) {
  const int num_ops = model_specs_[model_id].num_ops;
  const std::set<int>& unsupported_ops =
      model_specs_[model_id].unsupported_ops[device_flag];

  if (!planner_->NeedFallbackSubgraphs()) {
    return {{device_flag, {}}};
  }

  // TODO: Context-independent code / move to interpreter builder
  std::vector<DeviceOpIndices> subgraph_indices;
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
        kTfLiteCPUFallback : device_flag;

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
    const int model_id, std::set<DeviceOpIndices>& subgraph_indices) {
  if (!planner_->NeedFallbackSubgraphs()) {
    for (auto& worker : workers_) {
      TfLiteDeviceFlags device_flag = worker.first;
      subgraph_indices.insert({device_flag, {}});
    }
    return kTfLiteOk;
  }

  // Prepare variables to use
  const int num_ops = model_specs_[model_id].num_ops;
  Subgraph* primary_subgraph = subgraph(GetSubgraphIdx(model_id, kTfLiteCPU));

  // BitMask to check device support or not
  using BitMask = uint32_t;
  if (kTfLiteNumDevices > 8 * sizeof(BitMask)) {
    TFLITE_LOG(ERROR) << "kTfLiteNumDevices is larger than BitMask: "
                      << kTfLiteNumDevices;
  }

  // Build op_support_table
  std::vector<BitMask> op_support_table(num_ops, 0U);
  const std::map<TfLiteDeviceFlags, std::set<int>>& unsupported_ops =
      model_specs_[model_id].unsupported_ops;
  for (int op_index = 0; op_index < num_ops; op_index++) {
    for (auto& worker : workers_) {
      TfLiteDeviceFlags device_flag = worker.first;
      int device_id = device_flag;
      if (device_flag == kTfLiteCPU || device_flag == kTfLiteCPUFallback) {
        op_support_table[op_index] |= 1 << device_id;
        continue;
      }
      if (unsupported_ops.find(device_flag) == unsupported_ops.end() ||
          unsupported_ops.at(device_flag).find(op_index) == unsupported_ops.at(device_flag).end()) {
        op_support_table[op_index] |= 1 << device_id;
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
        subgraph_indices.insert({device_flag, unit_subgraph_ops});
      }
    }
  }
  if (!remaining_ops.empty()) {
    TFLITE_LOG(ERROR) << "Not empty remaining ops";
    return kTfLiteError;
  }
  return kTfLiteOk;
}

void Interpreter::InvestigateModelSpec(int model_id) {
  // get the subgraph index for this model
  int subgraph_index = GetSubgraphIdx(model_id, kTfLiteCPU);
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

    if (device_flag == kTfLiteCPU || device_flag == kTfLiteCPUFallback) {
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
    primary_subgraph->UndoAllDelegates();
  }

  primary_subgraph->AllocateTensors();
}

std::pair<int, int64_t> Interpreter::GetShortestLatency(
    int model_id, std::set<int> resolved_tensors, int64_t start_time,
    std::map<TfLiteDeviceFlags, int64_t>& device_waiting,
    int preceded_subgraph_index) {
  std::pair<int, std::set<int>> cache_key = {model_id, resolved_tensors};
  bool wait_time_is_stale = true;
  for (auto& pair : device_waiting) {
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
        GetShortestSubgraphIndex(iter->second, start_time, device_waiting);
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
                             device_waiting, target_subgraph.first);
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
    assert(cache_.find(cache_key) == cache_.end());
    assert(min_subgraph.second >= start_time);
    cache_[cache_key] = {min_subgraph.first,
                         min_subgraph.second - start_time};
  }

  return min_subgraph;
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
    std::vector<int> subgraph_indices, int64_t start_time,
    std::map<TfLiteDeviceFlags, int64_t>& device_waiting) {
  int64_t min_latency = INT_MAX;
  int min_idx = 0;

  for (auto subgraph_index : subgraph_indices) {
    SubgraphKey& key = subgraph(subgraph_index)->GetKey();

    int64_t waiting_time = device_waiting[key.device_flag];
    int64_t expected_latency = GetExpectedLatency(key);
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

    auto it = moving_averaged_latencies_.find(subgraph_key);
    if (it != moving_averaged_latencies_.end()) {
      int64_t latency = it->second;
      if (worst_latency < latency) {
        worst_latency = latency;
      }
    }
  }

  if (worst_latency == 0) {
    TFLITE_LOG(ERROR) << "Model #" << model_id << " has no profile results, "
                      << "but GetWorstDeviceProfileResult was called.";
  }

  return worst_latency;
}
}  // namespace impl

}  // namespace tflite
