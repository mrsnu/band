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
#include "tensorflow/lite/core/cpu/cpu.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/delegates/status.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tflite_with_xnnpack_optional.h"
#ifdef TFLITE_BUILD_WITH_XNNPACK_DELEGATE
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif
#include "tensorflow/lite/profiling/time_profiler.h"

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
                         TfLitePlannerType planner_type)
    : error_reporter_(error_reporter ? error_reporter :
                                       DefaultErrorReporter()) {
  // TODO(b/128420794): Include the TFLite runtime version in the log.
  // Prod logging is useful for mobile platforms where scraping console logs is
  // critical for debugging.
#if defined(TFLITE_IS_MOBILE_PLATFORM)
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#else
  TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#endif

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
  // FixedDevicePlanner is the default planner.
  planner_type_ = planner_type;
  if (planner_type == kRoundRobin) {
    planner_.reset(new RoundRobinPlanner(this));
  } else if (planner_type == kShortestExpectedLatency) {
    planner_.reset(new ShortestExpectedLatencyPlanner(this));
  } else {
    planner_.reset(new FixedDevicePlanner(this));
  }

  std::set<TfLiteDeviceFlags> valid_devices = { kTfLiteCPU, kTfLiteCPUFallback };

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
  // qualcomm hexagon : qti-default, qti-dsp, qti-gpu, qti-hta
  // google tpu: google-edgetpu
  // arm npu (DaVinci) : armnn
  // mediatek APU : neuron-ann
  for (const char* device_name : string_device_names_list) {
    if (IsNNAPIDeviceUseful(device_name)) {
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
    workers_[device_flag] = std::make_unique<Worker>(planner_, device_flag);
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

  // This essentially changes the "external_contexts_[type]".
  primary_subgraph().SetExternalContext(type, ctx);
}

TfLiteStatus Interpreter::SetInputs(std::vector<int> inputs) {
  return primary_subgraph().SetInputs(std::move(inputs));
}

TfLiteStatus Interpreter::SetOutputs(std::vector<int> outputs) {
  return primary_subgraph().SetOutputs(std::move(outputs));
}

TfLiteStatus Interpreter::SetVariables(std::vector<int> variables) {
  return primary_subgraph().SetVariables(std::move(variables));
}

TfLiteStatus Interpreter::AllocateTensors() {
  TfLiteStatus status;

  for (int i = 0; i < subgraphs_.size(); ++i) {
    status = (*subgraphs_[i]).AllocateTensors();
    if (status != kTfLiteOk)
      return status;
  }

  return kTfLiteOk;
}

void Interpreter::ReserveNodes(int count) {
  primary_subgraph().ReserveNodes(count);
}

void Interpreter::AddSubgraphs(int subgraphs_to_add,
                               int* first_new_subgraph_index) {
  const size_t base_index = subgraphs_.size();
  if (first_new_subgraph_index) *first_new_subgraph_index = base_index;

  subgraphs_.reserve(base_index + subgraphs_to_add);
  for (int i = 0; i < subgraphs_to_add; ++i) {
    Subgraph* subgraph = new Subgraph(error_reporter_, external_contexts_,
                                      &subgraphs_, &resources_);
    subgraph->SetProfiler(installed_profiler_, base_index + i);
    subgraphs_.emplace_back(subgraph);
  }

  // TODO #7: Change how the interpreter manages context of each subgraph
  context_ = primary_subgraph().context();
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
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  return primary_subgraph().AddNodeWithParameters(
      inputs, outputs, {}, init_data, init_data_size, builtin_data,
      registration, node_index);
}

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Interpreter::ResizeInputTensorStrict(
    int tensor_index, const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensorStrict(tensor_index, dims);
}

TfLiteStatus Interpreter::ReleaseNonPersistentMemory() {
  // TODO(b/138790287): We could do this for all subgraphs whose tensors have
  // been allocated. However, AllocateTensors() relies on Control Flow ops to
  // allocate tensors on 'children' subgraphs. Revisit this if required.
  return primary_subgraph().ReleaseNonPersistentMemory();
}

TfLiteStatus Interpreter::Invoke() {
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(installed_profiler_,
                                                           "invoke");
  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
      scoped_runtime_event, primary_subgraph().Invoke());

  if (!allow_buffer_handle_output_) {
    for (int tensor_index : outputs()) {
      TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
          scoped_runtime_event,
          primary_subgraph().EnsureTensorDataIsReadable(tensor_index));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Interpreter::Invoke(int idx) {
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(installed_profiler_,
                                                           "invoke");
  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
      scoped_runtime_event, (*subgraphs_[idx]).Invoke());

  if (!allow_buffer_handle_output_) {
    for (int tensor_index : (*subgraphs_[idx]).outputs()) {
      TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
          scoped_runtime_event,
          (*subgraphs_[idx]).EnsureTensorDataIsReadable(tensor_index));
    }
  }

  return kTfLiteOk;
}

void Interpreter::InvokeModelAsync(int model_id) {
  InvokeModelAsync(Job(model_id));
}
 
void Interpreter::InvokeModelAsync(Job request) {
  InvokeModelsAsync({request});
}

void Interpreter::InvokeModelsAsync() {
  std::vector<Job> requests;

  for (auto& m : model_configs_) {
    int model_id = m.first;
    ModelConfig& model_config = m.second;
    Job request = Job(model_id);
    for (int k = 0; k < model_config.batch_size; ++k) {
      requests.push_back(request);
    }
  }
  
  InvokeModelsAsync(requests);
}

void Interpreter::InvokeModelsAsync(std::vector<Job> requests) {
  if (requests.size() == 0) {
    return;
  }

  for (auto& request: requests) {
    int model_id = request.model_id;
    request.model_fname = model_configs_[model_id].model_fname;
    request.device_id = model_configs_[model_id].device;
  }

  planner_->EnqueueBatch(requests);
}

void Interpreter::InvokeModelsSync() {
  planner_->InitNumSubmittedJobs();
  InvokeModelsAsync();
  planner_->Wait();
}

void Interpreter::InvokeModelsSync(std::vector<Job> requests) {
  planner_->InitNumSubmittedJobs();
  InvokeModelsAsync(requests);
  planner_->Wait();
}

TfLiteStatus Interpreter::AddTensors(int tensors_to_add,
                                     int* first_new_tensor_index) {
  return primary_subgraph().AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::ResetVariableTensors() {
  return primary_subgraph().ResetVariableTensors();
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, dims.size(), dims.data(), quantization, buffer,
      bytes, allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    bool is_variable) {
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, dims.size(), dims.data(), quantization,
      is_variable);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, const char* buffer,
    size_t bytes, const Allocation* allocation) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, rank, dims, new_quantization, buffer, bytes,
      allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, rank, dims, new_quantization, is_variable,
      rank_dims_signature, dims_signature);
}

TfLiteStatus Interpreter::SetExecutionPlan(const std::vector<int>& new_plan) {
  return primary_subgraph().SetExecutionPlan(new_plan);
}

void Interpreter::UseNNAPI(bool enable) { primary_subgraph().UseNNAPI(enable); }

void Interpreter::SetNumThreads(int num_threads,
                                size_t first_subgraph_index,
                                int last_subgraph_index) {
  if (num_threads < -1) {
    // TODO #7 : Which context should we use here?
    context_->ReportError(context_,
                          "num_threads should be >=0 or just -1 to let TFLite "
                          "runtime set the value.");
    return;
  }

  if (last_subgraph_index < 0)
    last_subgraph_index = subgraphs_size();

  for (int i = first_subgraph_index; i < last_subgraph_index; i++) {
    subgraphs_[i]->context()->recommended_num_threads = num_threads;
  }

  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    auto* c = external_contexts_[i];
    if (c && c->Refresh) {
      c->Refresh(context_);
    }
  }
}

void Interpreter::SetXNNPACKNumThreads(int num_threads) {
  if (num_threads < -1) {
    // TODO #7 : Which context should we use here?
    context_->ReportError(context_,
                          "num_threads should be >=0 or just -1 to let TFLite "
                          "runtime set the value.");
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

// TODO(b/121264966): Subgraphs added after cancellation is set will not get the
// cancellation function added to their context.
void Interpreter::SetCancellationFunction(void* data,
                                          bool (*check_cancelled_func)(void*)) {
  for (auto& subgraph : subgraphs_) {
    subgraph->SetCancellationFunction(data, check_cancelled_func);
  }
}

bool Interpreter::IsCancelled() { return primary_subgraph().IsCancelled(); }

TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  TfLiteStatus status = kTfLiteOk;
  for (auto& subgraph : subgraphs_) {
    status = subgraph->ModifyGraphWithDelegate(delegate);
    if (status != kTfLiteOk) {
      break;
    }
  }
  // Delegate-specific errors can be recovered from by restoring Interpreter to
  // its original state.
  if (status == kTfLiteDelegateError) {
    TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
  }
  return status;
}

TfLiteStatus Interpreter::RemoveAllDelegates() {
  for (auto& subgraph : subgraphs_) {
    TF_LITE_ENSURE_STATUS(subgraph->RemoveAllDelegates());
  }
  return kTfLiteOk;
}

bool Interpreter::HasDelegates() { return primary_subgraph().HasDelegates(); }

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  std::vector<TfLiteTensor>& tensors = primary_subgraph().tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  TF_LITE_ENSURE(context_,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE(context_, tensor->delegate->FreeBufferHandle != nullptr);
    tensor->delegate->FreeBufferHandle(context_, tensor->delegate,
                                       &tensor->buffer_handle);
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  std::vector<TfLiteTensor>& tensors = primary_subgraph().tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

void Interpreter::Profile(const int num_warm_ups, const int num_runs,
                          ModelDeviceToLatency& profiled) {
  tflite::Profiler* previous_profiler = GetProfiler();
  // Assign temporal time profiler for profiling.
  tflite::profiling::TimeProfiler timer;
  // Only update subgraph profilers to not care ownership of the profiler
  SetSubgraphProfiler(&timer);

  for (int i = 0; i < subgraphs_size(); ++i) {
    Subgraph* subgraph = subgraphs_[i].get();
    SubgraphKey& subgraph_key = subgraph->GetKey();

    auto it = profiled.find(subgraph_key);
    if (it != profiled.end()) {
      // if an entry for this SubgraphKey exists in the profiled data,
      // then reuse it to reduce initialization time
      int64_t profiled_latency = it->second;
      subgraph_profiling_results_map_[subgraph_key] = profiled_latency;

      error_reporter_->Report("Reusing profiled result\n model=%d avg=%d us device=%s start=%d end=%d.",
          subgraph_key.model_id, profiled_latency,
          TfLiteDeviceGetName(subgraph_key.device_flag),
          subgraph_key.start_idx, subgraph_key.end_idx);

    } else {
      // otherwise, proceed as normal
      for (int i = 0; i < num_warm_ups; i++) {
        subgraph->Invoke();
      }
      timer.ClearRecords();
      for (int i = 0; i < num_runs; i++) {
        subgraph->Invoke();
      }

      int64_t latency = timer.GetAverageElapsedTime<std::chrono::microseconds>();
      subgraph_profiling_results_map_[subgraph_key] = latency;

      // record the profiled latency for subsequent benchmark runs
      profiled[subgraph_key] = latency;

      error_reporter_->Report("Profiling result\n model=%d warmup=%d count=%d avg=%d us device=%s start=%d end=%d.",
              subgraph_key.model_id, num_warm_ups, num_runs, latency,
              TfLiteDeviceGetName(subgraph_key.device_flag),
              subgraph_key.start_idx, subgraph_key.end_idx);
    }
  }

  SetSubgraphProfiler(previous_profiler);
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
  for (int subgraph_index = 0; subgraph_index < subgraphs_.size();
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

TfLiteStatus Interpreter::PrepareLogging(std::string log_path) {
  if (!planner_)
    return kTfLiteError;
  return planner_->PrepareLogging(log_path);
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

void Interpreter::RegisterSubgraphIdx(SubgraphKey subgraph_key,
                                      int subgraph_idx) {
  // Skip if exists.
  if (subgraph_idx_map_.find(subgraph_key) != subgraph_idx_map_.end()) {
    return;
  }
  subgraph_idx_map_[subgraph_key] = subgraph_idx;
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
    error_reporter_->Report("ERROR: Cannot find the worker.");
    // TODO #21: Handle errors in multi-thread environment
    return nullptr;
  }
}

int Interpreter::GetSubgraphIdx(int model_id, int device_idx) {
  TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_idx);
  return GetSubgraphIdx(model_id, device_flag);
}

int Interpreter::GetSubgraphIdx(int model_id, TfLiteDeviceFlags device_id) {
  // start_idx and end_idx weren't specified, so we assume that the caller
  // intended to retrieve the whole model
  ModelSpec& spec = model_specs_[model_id];
  SubgraphKey key(model_id, device_id, 0, spec.num_ops - 1);
  auto it = subgraph_idx_map_.find(key);
  if (it != subgraph_idx_map_.end()) {
    return it->second;
  } else {
    return -1;
  }
}

std::set<int> Interpreter::models() const {
  std::set<int> models;
  for (auto& key : subgraph_idx_map_) {
    models.insert(key.first.model_id);
  }
  return models;
}

TfLiteStatus Interpreter::SetWorkerThreadAffinity(const CpuSet& thread_affinity_mask, TfLiteDeviceFlags device_id) {
  if (device_id == kTfLiteNumDevices) {
    for (auto& deviceWorker : workers_) {
      if (deviceWorker.second->SetWorkerThreadAffinity(thread_affinity_mask) != kTfLiteOk)
        return kTfLiteError;
    }
    return kTfLiteOk;
  } else {
    return workers_[device_id]->SetWorkerThreadAffinity(thread_affinity_mask);
  }
}

int64_t Interpreter::GetDeviceWaitingTime(TfLiteDeviceFlags device) {
  int64_t waiting_time = 0;
  Worker* worker = GetWorker(device);
  {
    std::lock_guard<std::mutex> lock(worker->GetDeviceMtx());
    std::deque<Job>& requests = worker->GetDeviceRequests();
    for (auto& job : requests) {
      SubgraphKey subgraph_key(job.model_id, device, job.start_idx, job.end_idx);
      // TODO (dhkim): what if no valid subgraph key is given?
      waiting_time += GetSubgraphProfileResult(subgraph_key);
    }
  }

  return waiting_time;
}

int64_t Interpreter::GetSubgraphProfileResult(SubgraphKey& key) {
  auto it = subgraph_profiling_results_map_.find(key);
  if (it != subgraph_profiling_results_map_.end()) {
    return it->second;
  } else {
    return -1;
  }
}

void Interpreter::MakeSubgraphsForFallbackOps(const int model_id,
                                              const TfLiteDeviceFlags device_flag,
                                              std::vector<SubgraphKey>& splitted_op_range) {
  const int num_ops = model_specs_[model_id].num_ops;
  const std::vector<int>& unsupported_ops =
      model_specs_[model_id].unsupported_ops[device_flag];

  TfLiteDeviceFlags curr_device = device_flag;
  TfLiteDeviceFlags prev_device;
  int subgraph_min = 0;

  if (planner_type_ == kFixedDevice) {
    splitted_op_range.push_back(SubgraphKey(model_id, device_flag, 0, num_ops - 1));
    return;
  }

  // do a pass through all ops and create a subgraph every time the supported
  // device changes, using `subgraph_min` to keep track of the current
  // subgraph's starting op
  for (int k = 0; k < num_ops; ++k) {
    prev_device = curr_device;

    // check if this ops is supported by `device_flag` or not
    if (std::find(unsupported_ops.begin(), unsupported_ops.end(), k)
        != unsupported_ops.end()) {
      curr_device = kTfLiteCPUFallback;
    } else {
      curr_device = device_flag;
    }

    // if the current op is supported while the prev op is unsupported
    // (or vice versa), then make a subgraph up until the prev op
    if (k > 0 && curr_device != prev_device) {
      splitted_op_range.push_back(SubgraphKey(model_id, prev_device,
                                              subgraph_min, k - 1));
      subgraph_min = k;
    }

    // if this is the last op, then there are no more ops to check so
    // register the final subgraph
    if (k == num_ops - 1) {
      splitted_op_range.push_back(SubgraphKey(model_id, curr_device,
                                              subgraph_min, num_ops - 1));
    }
  }
}

int Interpreter::GetWindowSize() const {
  return planner_->GetWindowSize();
}

void Interpreter::SetWindowSize(int schedule_window_size) {
  planner_->SetWindowSize(schedule_window_size);
}

void Interpreter::AllowWorkSteal() {
  for (auto& worker : workers_) {
    worker.second->AllowWorkSteal();
  }
}

void Interpreter::InvestigateModelSpec(int model_id) {
  // get the subgraph index for this model
  // at this point, the subgraph key for this model doesn't have valid start
  // and end indices so we don't need to specify them
  int subgraph_idx = GetSubgraphIdx(SubgraphKey(model_id, kTfLiteCPU));
  Subgraph* primary_subgraph = subgraph(subgraph_idx);

  // this creates an empty ModelSpec
  ModelSpec& model_spec = model_specs_[model_id];

  std::vector<int>& execution_plan = primary_subgraph->execution_plan();
  model_spec.num_ops = execution_plan.size();

  // delete the current key and replace it with valid start/end indices
  SubgraphKey& key = primary_subgraph->GetKey();
  DeleteKey(key);
  key.start_idx = 0;
  key.end_idx = model_spec.num_ops - 1;
  RegisterSubgraphIdx(key, subgraph_idx);

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
      model_spec.output_tensors.insert(output_tensor);
    }

    for (auto i : tensor_indices) {
      const auto* tensor = primary_subgraph->tensor(i);
      model_spec.tensor_types.insert(tensor->type);
    }
  }

  // also check unsupported ops to fill in model_spec.unsupported_ops
  for (int i = 0; i < kTfLiteNumDevices; ++i) {
    TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(i);

    if (device_flag == kTfLiteCPU) {
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
        model_spec.unsupported_ops[device_flag].push_back(node_index);
      }
    }

    // revert changes
    primary_subgraph->UndoAllDelegates();
  }

  primary_subgraph->AllocateTensors();
}

std::pair<int, int64_t>
Interpreter::GetShortestLatency(int model_id, int start_idx, int64_t start_time,
                                std::vector<int64_t>& device_waiting,
                                TfLiteDeviceFlags preceded_device) {
  std::vector<int> subgraph_indices = GetSubgraphCandidates(model_id, start_idx,
                                                            preceded_device);
  std::map<std::string, std::vector<int>> subgraph_map =
      GroupByStartEndIdx(subgraph_indices);

  std::pair<int, int64_t> min_subgraph = {-1, INT_MAX};
  for (auto iter = subgraph_map.begin(); iter != subgraph_map.end() ; iter++) {
    // first, filter out the subgraphs that take longer than others with the
    // same start/end indices, since there's no reason to pick them
    std::pair<int, int64_t> target_subgraph =
        GetShortestSubgraphIndex(iter->second, start_time, device_waiting);
    SubgraphKey& key = subgraph(target_subgraph.first)->GetKey();

    std::pair<int, int64_t> local_min;
    if (key.end_idx != model_specs_[model_id].num_ops - 1) {
      // there's more ops left for this model, so we need to look further to
      // get the final latency
      local_min = GetShortestLatency(model_id, key.end_idx + 1,
                                     target_subgraph.second, device_waiting,
                                     key.device_flag);
    } else {
      // nothing else after this
      local_min = target_subgraph;
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
  return min_subgraph;
}


std::map<std::string, std::vector<int>> Interpreter::GroupByStartEndIdx(std::vector<int> subgraph_indices) {
  std::map<std::string, std::vector<int>> ret;
  for (auto subgraph_idx : subgraph_indices) {
    SubgraphKey& subgraph_key = subgraph(subgraph_idx)->GetKey();
    std::string key = std::to_string(subgraph_key.start_idx) +
                      "/" +
                      std::to_string(subgraph_key.end_idx);
    ret[key].push_back(subgraph_idx);
  }
  return ret;
}

std::vector<int> Interpreter::GetSubgraphCandidates(int model_id, int start_idx,
                                                    TfLiteDeviceFlags preceded_device) {
  std::vector<int> candidatesIds;
  // iterate thru all subgraphs and only pick the ones that match the criteria
  for (int i = 0; i < subgraphs_size(); ++i) {
    SubgraphKey& key = subgraph(i)->GetKey();
    if (key.model_id == model_id &&
        key.start_idx == start_idx &&
        key.device_flag != preceded_device) {
      candidatesIds.push_back(i);
    }
  }
  return candidatesIds;
}

std::pair<int, int64_t>
Interpreter::GetShortestSubgraphIndex(std::vector<int> subgraph_indices,
                                      int64_t start_time,
                                      std::vector<int64_t>& device_waiting) {
  int64_t min_latency = INT_MAX;
  int min_idx = 0;

  for (auto subgraph_idx : subgraph_indices) {
    SubgraphKey& key = subgraph(subgraph_idx)->GetKey();

    int64_t waiting_time = device_waiting[key.device_flag];
    int64_t profiled = GetSubgraphProfileResult(key);
    int64_t expected_latency = profiled + std::max(waiting_time, start_time);

    if (min_latency > expected_latency) {
      min_latency = expected_latency;
      min_idx = subgraph_idx;
    }
  }
  return { min_idx, min_latency };
}

}  // namespace impl

}  // namespace tflite
