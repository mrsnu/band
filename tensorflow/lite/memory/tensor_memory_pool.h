#ifndef TENSORFLOW_LITE_MEMORY_TENSOR_MEMORY_POOL_H_
#define TENSORFLOW_LITE_MEMORY_TENSOR_MEMORY_POOL_H_

#include <array>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/memory/simple_memory_pool.h"
#include "tensorflow/lite/util.h"

namespace tflite {
class ErrorReporter;
class TensorMemoryPool : public SimpleMemoryPool<SimpleMemoryPool<MemoryBlock>> {
 public:
  TensorMemoryPool(ErrorReporter* error_reporter, size_t size_bytes = 1 << 16,
                   size_t tensor_alignment = kTfLiteTensorDefaultAlignment);
  ~TensorMemoryPool();

  TfLiteStatus GetTensorFromHandle(TfLiteTensor* dst_tensor, int job_id, int tensor_id);
  TfLiteStatus PutTensorToHandle(TfLiteTensor* src_tensor, int job_id, int tensor_id);

  virtual TfLiteStatus Allocate(size_t size_bytes, int job_id);
  virtual TfLiteStatus Deallocate(int job_id);

 private:
  virtual TfLiteStatus Resize(size_t size_bytes);
  TfLiteStatus EnsureHandleTensors(const Tensors& tensors, int handle,
                                   bool check_current = false) const;

  std::mutex head_handle_mtx_;
  mutable std::mutex buffer_mtx_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_MEMORY_TENSOR_MEMORY_POOL_H_
