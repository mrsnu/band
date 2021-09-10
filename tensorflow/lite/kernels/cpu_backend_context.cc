/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/cpu_backend_context.h"

#include <memory>

#include "public/gemmlowp.h"
#include "ruy/context.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/tools/logging.h"

namespace {
const int kDefaultNumThreadpoolThreads = 1;

}  // namespace

namespace tflite {

CpuBackendContext* CpuBackendContext::GetFromContext(TfLiteContext* context) {
  auto* external_context = static_cast<ExternalCpuBackendContext*>(
      context->GetExternalContext(context, kTfLiteCpuBackendContext));

  if (external_context == nullptr) {
    TF_LITE_FATAL(
        "ExternalCpuBackendContext isn't properly initialized during TFLite "
        "interpreter initialization.");
  }

  auto* cpu_backend_context = static_cast<CpuBackendContext*>(
      external_context->internal_backend_context());
  if (cpu_backend_context == nullptr) {
    // We do the lazy initialization here for the TfLiteInternalBackendContext
    // that's wrapped inside ExternalCpuBackendContext.
    cpu_backend_context = new CpuBackendContext();
    cpu_backend_context->SetMaxNumThreads(context->recommended_num_threads);
    external_context->set_internal_backend_context(
        std::unique_ptr<TfLiteInternalBackendContext>(cpu_backend_context));
  }

  return cpu_backend_context;
}

CpuBackendContext::CpuBackendContext()
    : TfLiteInternalBackendContext(),
      gemmlowp_context_(new gemmlowp::GemmContext) {
  SetMaxNumThreads(kDefaultNumThreadpoolThreads);
// TODO(b/148289189) Remove when clients have transitioned to runtime flag.
#ifdef TFLITE_WITH_RUY_GEMV
  SetUseCaching(true);
#else
  SetUseCaching(false);
#endif
}

CpuBackendContext::~CpuBackendContext() {}

void CpuBackendContext::SetMaxNumThreads(int max_num_threads) {
  const int target_num_threads =
      max_num_threads > -1 ? max_num_threads : kDefaultNumThreadpoolThreads;
  max_num_threads_ = target_num_threads;
  for (auto& pair : ruy_contexts_) {
    pair.second->set_max_num_threads(target_num_threads);
  }
  gemmlowp_context_->set_max_num_threads(target_num_threads);
}

void CpuBackendContext::SetCpuSet(std::thread::id tid, impl::CpuSet cpu_mask) {
  cpu_masks_.insert(std::make_pair(tid, cpu_mask));
  UpdateCpuSet(tid);
}

void CpuBackendContext::SetUseCaching(bool flag) { use_caching_ = flag; }

ruy::Context* CpuBackendContext::ruy_context() {
  std::thread::id this_id = std::this_thread::get_id();
  std::lock_guard<std::mutex> lock(ruy_context_lock_);
  if (ruy_contexts_.find(this_id) == ruy_contexts_.end()) {
    ruy_contexts_[this_id] = std::make_unique<ruy::Context>();
    UpdateCpuSet(this_id);
  }
  return ruy_contexts_[this_id].get();
}

void CpuBackendContext::ClearCaches() {
  for (auto& pair : ruy_contexts_) {
    pair.second->ClearPrepackedCache();
  }
}

void CpuBackendContext::UpdateCpuSet(std::thread::id tid) {
  if (ruy_contexts_.find(tid) != ruy_contexts_.end() &&
      cpu_masks_.find(tid) != cpu_masks_.end()) {
    impl::CpuSet current_set = cpu_masks_[tid];
    int max_threads = std::min(max_num_threads_, current_set.NumEnabled());
    ruy_contexts_[tid]->set_max_num_threads(max_threads);
    ruy_contexts_[tid]->set_cpu_mask(current_set.GetCpuSet().__bits);
    TFLITE_LOG(INFO) << "Ruy tid " << tid << " number of threads " << max_threads;
  }
}

}  // namespace tflite
