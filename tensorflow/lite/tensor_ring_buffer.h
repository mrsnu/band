#ifndef TENSORFLOW_LITE_RING_BUFFER_H_
#define TENSORFLOW_LITE_RING_BUFFER_H_

#include <array>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

class ErrorReporter;
class TensorRingBuffer {
 public:
  TensorRingBuffer(ErrorReporter* error_reporter, std::vector<TfLiteTensor*> tensors, size_t size = 32);
  ~TensorRingBuffer();

  size_t Alloc();
  bool IsValid(size_t handle) const;
  const std::vector<TfLiteTensor>* Get(size_t handle) const;
  TfLiteStatus Put(std::vector<TfLiteTensor*> tensors, size_t handle);

 private:
  size_t GetIndex(size_t handle) const;

  size_t head_ = 0;
  size_t size_;
  std::vector<TfLiteTensor>* tensors_;
  ErrorReporter* error_reporter_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_RING_BUFFER_H_