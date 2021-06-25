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
  TensorRingBuffer(ErrorReporter* error_reporter, Tensors tensors, int size = 64);
  ~TensorRingBuffer();

  int Alloc();
  bool IsValid(int handle) const;
  const Tensors* Get(int handle) const;
  TfLiteStatus Put(const Tensors& tensors, int handle);

 private:
  int GetIndex(int handle) const;

  int head_ = 0;
  int size_;
  std::vector<TfLiteTensor*>* tensors_;
  ErrorReporter* error_reporter_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_RING_BUFFER_H_