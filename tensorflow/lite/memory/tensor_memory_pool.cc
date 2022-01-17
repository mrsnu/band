#include "tensorflow/lite/memory/tensor_memory_pool.h"

#include <cassert>

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
TensorMemoryPool::TensorMemoryPool(ErrorReporter* error_reporter,
                                   size_t size_bytes, size_t tensor_alignment)
    : SimpleMemoryPool<SimpleMemoryPool<MemoryBlock>>(
          error_reporter, size_bytes, tensor_alignment) {}

TensorMemoryPool::~TensorMemoryPool() {}

TfLiteStatus TensorMemoryPool::GetTensorFromHandle(TfLiteTensor* dst_tensor,
                                                   int job_id, int tensor_idx) {
  // get memory pool of job
  auto job_tensor_pool_it = handle_offsets_.find(job_id);
  TF_LITE_ENSURE_NEQ(error_reporter_, job_tensor_pool_it,
                     handle_offsets_.end());
  auto& job_tensor_pool = job_tensor_pool_it->second;
  // get tensor buffer of job
  char* tensor_buffer = job_tensor_pool.GetBufferFromHandle(tensor_idx);
  TF_LITE_ENSURE_NEQ(error_reporter_, tensor_buffer, nullptr);
  memcpy(dst_tensor->data.raw, tensor_buffer, dst_tensor->bytes);
  TF_LITE_ENSURE_OK(error_reporter_,
                    job_tensor_pool_it->second.Deallocate(tensor_idx));
  return kTfLiteOk;
}

TfLiteStatus TensorMemoryPool::PutTensorToHandle(TfLiteTensor* src_tensor,
                                                 int job_id, int tensor_idx) {
  auto job_tensor_pool_it = handle_offsets_.find(job_id);
  TF_LITE_ENSURE_NEQ(error_reporter_, job_tensor_pool_it,
                     handle_offsets_.end());
  // allocate tensor fron memory pool of job
  auto& job_tensor_pool = job_tensor_pool_it->second;
  TF_LITE_ENSURE(
      error_reporter_,
      job_tensor_pool.Allocate(src_tensor->bytes, tensor_idx) == kTfLiteOk);
  memcpy(job_tensor_pool.GetBufferFromHandle(tensor_idx), src_tensor->data.raw,
         src_tensor->bytes);
  return kTfLiteOk;
}

TfLiteStatus TensorMemoryPool::Allocate(size_t size_bytes, int job_id) {
  std::lock_guard<std::mutex> lock(head_handle_mtx_);
  return SimpleMemoryPool<SimpleMemoryPool<MemoryBlock>>::Allocate(size_bytes,
                                                                   job_id);
}

TfLiteStatus TensorMemoryPool::Deallocate(int job_id) {
  std::lock_guard<std::mutex> lock(head_handle_mtx_);
  return SimpleMemoryPool<SimpleMemoryPool<MemoryBlock>>::Deallocate(job_id);
}

TfLiteStatus TensorMemoryPool::Resize(size_t size_bytes) {
  std::lock_guard<std::mutex> lock(buffer_mtx_);
  return SimpleMemoryPool<SimpleMemoryPool<MemoryBlock>>::Resize(size_bytes);
}
}  // namespace tflite