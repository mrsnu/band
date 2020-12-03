/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/mixed/mixed_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/profiling/time.h"

#include <string>
#include <vector>

#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {

using namespace tflite::gpu;

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

// An MixedDelegateWrapper is responsibile to manage a TFLite delegate
// initialized from a shared library. It creates a delegate from the given
// option and storages it to mixed_delegate_ member variable. On the
// destruction, it conducts necessary clean up process.
class MixedDelegateWrapper {
 public:
  explicit MixedDelegateWrapper(
      const TfLiteMixedDelegateOptions* options);
  ~MixedDelegateWrapper();

  // Return a TfLiteDelegate which is convertibile to this class.
  TfLiteDelegate* tflite_wrapper_delegate() { return &wrapper_delegate_; }
  TfLiteDelegate* tflite_gpu_delegate() { return gpu_delegate_; }
  TfLiteDelegate* tflite_nnapi_delegate() { return nnapi_delegate_; }

 private:
 TfLiteDelegate wrapper_delegate_ = {
          .data_ = reinterpret_cast<void*>(this),
          .Prepare = DelegatePrepare,
          .CopyFromBufferHandle = nullptr,
          .CopyToBufferHandle = nullptr,
          .FreeBufferHandle = nullptr,
          .flags = kTfLiteDelegateFlagsNone,
      };
 
 TfLiteDelegate* nnapi_delegate_;
 TfLiteDelegate* gpu_delegate_;
};

// Converts the given TfLiteDelegate to an MixedDelegateWrapper instance.
inline MixedDelegateWrapper* GetMixedDelegateWrapper(
    TfLiteDelegate* delegate) {
  return reinterpret_cast<MixedDelegateWrapper*>(delegate->data_);
}

// Relay Prepare() call to the associated mixed TfLiteDelegate object.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  auto mixed_delegate_wrapper = GetMixedDelegateWrapper(delegate);
  TfLiteDelegate* mixed_delegate =
      mixed_delegate_wrapper->tflite_wrapper_delegate();
  auto* gpu_delegate = mixed_delegate_wrapper->tflite_gpu_delegate();
  auto* nnapi_delegate = mixed_delegate_wrapper->tflite_nnapi_delegate();

  TfLiteStatus status = kTfLiteOk;
  TfLiteIntArray* device_plan;
  context->GetDevicePlan(context, &device_plan);
  TfLiteIntArray* gpu_supported_ops = GetOpsToReplace(context, true);
  std::vector<int> dsp_supported_op_vector;
  reinterpret_cast<StatefulNnApiDelegate*>(nnapi_delegate)->GetSupportedNodes(context, nnapi_delegate, dsp_supported_op_vector);
	auto dsp_supported_ops = BuildTfLiteIntArray(dsp_supported_op_vector);

  std::vector<int> supported_device(device_plan->size, 0);
  for(int i = 0; i < gpu_supported_ops->size; ++i){
    supported_device[gpu_supported_ops->data[i]] |= 1;
  }
  for(int i = 0; i < dsp_supported_ops.get()->size; ++i){
    supported_device[dsp_supported_ops.get()->data[i]] |= 2;
  }

  int64_t start = profiling::time::NowMicros();
  for(int i = 0; i < device_plan->size; ++i){
    if(device_plan->data[i] & supported_device[i]) continue;
    else device_plan->data[i] = 0;
  }

  // from the end of the vector, replace continous ops to a delegate
  for(int i = device_plan->size - 1; i >= 0; --i){
    int current_device = device_plan->data[i];
    int end = i;
    int start = i - 1;
    while(current_device == device_plan->data[start] && start >= 0){
      start--;
    }

    i = start + 1;
    if(current_device == 0) continue;
    if(end - start <= 1) continue;

    std::vector<int> to_replace;
    for(int j = start + 1; j <= end; ++j){
      to_replace.push_back(j);
    }

	  auto to_replace_array = BuildTfLiteIntArray(to_replace);

    //GPU
    if(current_device == 1){
      status = context->ReplaceNodeSubsetsWithDelegateKernels(
          context, GetRegistration(), to_replace_array.get(), gpu_delegate);
    }
    //DSP
    if(current_device == 2){
      status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, reinterpret_cast<StatefulNnApiDelegate*>(nnapi_delegate)->nnapi_delegate_kernel, to_replace_array.get(), nnapi_delegate);
    }

    //TfLiteIntArrayFree(to_replace_array.get());
  }

  int64_t end = profiling::time::NowMicros();

  std::cout << "REPLACENODE TIME : " << end - start << " (us)" << std::endl;
  //TfLiteIntArrayFree(device_plan);

    /*
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  std::vector<int> nodes_to_replace;
  reinterpret_cast<StatefulNnApiDelegate*>(nnapi_delegate)->GetSupportedNodes(context, nnapi_delegate, nodes_to_replace);

	std::vector<int> nnapi_nodes;
	for(int i = 0; i < nodes_to_replace.size(); ++i){
		bool exists = false;
		for(int j = 0; j < ops_to_replace->size; ++j){
			if(ops_to_replace->data[j] == nodes_to_replace[i]){
				exists = true;
				j = ops_to_replace->size;
			}
		}
		if(!exists){
			nnapi_nodes.push_back(nodes_to_replace[i]);
		}
	}

  if(nodes_to_replace.size() == 0){
	status = kTfLiteOk;
  }
  else{
	auto supported_nodes_int_array = BuildTfLiteIntArray(nnapi_nodes);
	status = context->ReplaceNodeSubsetsWithDelegateKernels(
	context, reinterpret_cast<StatefulNnApiDelegate*>(nnapi_delegate)->nnapi_delegate_kernel, supported_nodes_int_array.get(), nnapi_delegate);
  }
  status = context->ReplaceNodeSubsetsWithDelegateKernels(
	 	  context, GetRegistration(), ops_to_replace, gpu_delegate);

  TfLiteIntArrayFree(ops_to_replace);
  */

  return status;
}

// Relay CopyFromBufferHandle() call to the associated mixed TfLiteDelegate
// object.
TfLiteStatus DelegateCopyFromBufferHandle(TfLiteContext* context,
                                          struct TfLiteDelegate* delegate,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteTensor* tensor) {
  auto mixed_delegate_wrapper = GetMixedDelegateWrapper(delegate);
  TfLiteDelegate* mixed_delegate =
      mixed_delegate_wrapper->tflite_gpu_delegate();
  return mixed_delegate->CopyFromBufferHandle(context, delegate,
                                                 buffer_handle, tensor);
}

// Relay CopyToBufferHandle() call to the associated mixed TfLiteDelegate
// object.
TfLiteStatus DelegateCopyToBufferHandle(TfLiteContext* context,
                                        struct TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) {
  auto mixed_delegate_wrapper = GetMixedDelegateWrapper(delegate);
  TfLiteDelegate* mixed_delegate =
      mixed_delegate_wrapper->tflite_gpu_delegate();
  return mixed_delegate->CopyToBufferHandle(context, delegate, buffer_handle,
                                               tensor);
}

// Relay FreeBufferHandle() call to the associated mixed TfLiteDelegate
// object.
void DelegateFreeBufferHandle(TfLiteContext* context,
                              struct TfLiteDelegate* delegate,
                              TfLiteBufferHandle* handle) {
  auto mixed_delegate_wrapper = GetMixedDelegateWrapper(delegate);
  TfLiteDelegate* mixed_delegate =
      mixed_delegate_wrapper->tflite_gpu_delegate();
  return mixed_delegate->FreeBufferHandle(context, delegate, handle);
}

StatefulNnApiDelegate::Options GetDefaultNnApiOptions(){
  StatefulNnApiDelegate::Options options = StatefulNnApiDelegate::Options();
  return options;
}

MixedDelegateWrapper::MixedDelegateWrapper(
    const TfLiteMixedDelegateOptions* options) {
	TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
	gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
	gpu_delegate_ = TfLiteGpuDelegateV2Create(&gpu_opts);
  gpu_delegate_->name = "GPU";
	StatefulNnApiDelegate::Options nnapi_options = GetDefaultNnApiOptions();
	nnapi_options.accelerator_name = "qti-dsp";
	nnapi_options.max_number_delegated_partitions = 15;
	//nnapi_delegate_ = reinterpret_cast<TfLiteDelegate*>(StatefulNnApiDelegate::StatefulNnApiDelegate(nnapi_options));
	nnapi_delegate_ = new StatefulNnApiDelegate(nnapi_options);
  nnapi_delegate_->name = "DSP";

  wrapper_delegate_.CopyFromBufferHandle = DelegateCopyFromBufferHandle;
  wrapper_delegate_.CopyToBufferHandle = DelegateCopyToBufferHandle;
  wrapper_delegate_.FreeBufferHandle = DelegateFreeBufferHandle;
}

MixedDelegateWrapper::~MixedDelegateWrapper() {
}

}  // namespace
}  // namespace tflite

// TfLiteMixedDelegateOptionsInsert adds key/value to the given
// TfLiteMixedDelegateOptions instance.
TfLiteStatus TfLiteMixedDelegateOptionsInsert(
    TfLiteMixedDelegateOptions* options, const char* key,
    const char* value) {
  if (options->count >= kMaxOptions) {
    return kTfLiteError;
  }
  options->keys[options->count] = key;
  options->values[options->count] = value;
  options->count++;
  return kTfLiteOk;
}

TfLiteMixedDelegateOptions TfLiteMixedDelegateOptionsDefault(
    const char* lib_path) {
  // As 'keys' and 'values' don't need to be set here, using designated
  // initializers may cause a compiling error as "non-trivial designated
  // initializers not supported" by some compiler.
  TfLiteMixedDelegateOptions options;
  options.lib_path = lib_path;
  options.count = 0;
  options.insert = TfLiteMixedDelegateOptionsInsert;
  return options;
}

TfLiteDelegate* TfLiteMixedDelegateCreate(
    const TfLiteMixedDelegateOptions* options) {
  auto* mixed_delegate_wrapper =
      new tflite::MixedDelegateWrapper(options);
  if (mixed_delegate_wrapper) {
    return mixed_delegate_wrapper->tflite_wrapper_delegate();
  }
  return nullptr;
}

void TfLiteMixedDelegateDelete(TfLiteDelegate* delegate) {
  delete tflite::GetMixedDelegateWrapper(delegate);
}
