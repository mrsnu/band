#ifndef TENSORFLOW_LITE_RING_BUFFER_H_
#define TENSORFLOW_LITE_RING_BUFFER_H_

#include <array>
#include <vector>

#include "tensorflow/lite/util.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

class ErrorReporter;
class TensorRingBuffer {
 public:
  TensorRingBuffer(ErrorReporter* error_reporter, std::vector<const TfLiteTensor*> tensors, int size = 64);
  ~TensorRingBuffer();

  int Alloc();
  bool IsValid(int handle) const;
  const std::vector<TfLiteTensor*>* Get(int handle) const;
  TfLiteStatus Put(const std::vector<TfLiteTensor*>& tensors, int handle);

 private:
  int GetIndex(int handle) const;

  int head_ = 0;
  int size_;
  std::vector<TfLiteTensor*>* tensors_;
  ErrorReporter* error_reporter_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_RING_BUFFER_H_