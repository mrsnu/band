#ifndef TENSORFLOW_LITE_RING_BUFFER_H_
#define TENSORFLOW_LITE_RING_BUFFER_H_

#include <array>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

class ErrorReporter;
class TensorRingBuffer {
 public:
  TensorRingBuffer(ErrorReporter* error_reporter, std::vector<const TfLiteTensor*> tensors, size_t size = 64);
  ~TensorRingBuffer();

  int Alloc();
  bool IsValid(int handle) const;
  const std::vector<TfLiteTensor>* Get(int handle) const;
  TfLiteStatus Put(std::vector<TfLiteTensor> tensors, int handle);

 private:
  size_t GetIndex(int handle) const;

  int head_ = 0;
  size_t size_;
  std::vector<TfLiteTensor>* tensors_;
  ErrorReporter* error_reporter_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_RING_BUFFER_H_