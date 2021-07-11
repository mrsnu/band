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
  TensorRingBuffer(ErrorReporter* error_reporter, Tensors tensors,
                   std::vector<int> tensor_indices, int size = 128);
  ~TensorRingBuffer();

  const int GetTensorsLength() const;
  int Alloc();
  bool IsTensorIndexValid(int tensor_index) const;
  bool IsHandleValid(int handle) const;
  TfLiteStatus GetTensorFromHandle(TfLiteTensor* dst, int tensor_index, int handle) const;
  TfLiteStatus PutTensorToHandle(const TfLiteTensor* src, int tensor_index, int handle);
  TfLiteStatus GetTensorsFromHandle(Tensors& dst_tensors, int handle) const;
  TfLiteStatus PutTensorsToHandle(const Tensors& src_tensors, int handle);

 private:
  int GetIndex(int handle) const;
  TfLiteStatus CopyTensors(const Tensors& src_tensors, Tensors& dst_tensors) const;
  TfLiteStatus CopyTensor(const TfLiteTensor* src, TfLiteTensor* dst) const;

  mutable std::mutex head_mtx_;
  int head_ = 0;
  const int size_;
  std::vector<TfLiteTensor*>* tensors_;
  // Tensor's model index to ring buffer's index
  std::map<int, int> model_to_buffer_;
  ErrorReporter* error_reporter_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_TENSOR_RING_BUFFER_H_
