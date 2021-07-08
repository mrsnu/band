#ifndef TENSORFLOW_LITE_TENSOR_RING_BUFFER_H_
#define TENSORFLOW_LITE_TENSOR_RING_BUFFER_H_

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
  TfLiteStatus GetTensorsFromHandle(Tensors& dst_tensors, int handle) const;
  TfLiteStatus PutTensorsToHandle(const Tensors& src_tensors, int handle);

 private:
  int GetIndex(int handle) const;
  TfLiteStatus CopyTensors(const Tensors& src_tensors, Tensors& dst_tensors) const;

  mutable std::mutex head_mtx_;
  int head_ = 0;
  int size_;
  std::vector<TfLiteTensor*>* tensors_;
  ErrorReporter* error_reporter_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_TENSOR_RING_BUFFER_H_
