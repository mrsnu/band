#ifndef BAND_TENSOR_RING_BUFFER_H_
#define BAND_TENSOR_RING_BUFFER_H_

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "band/c/common.h"
#include "band/interface/tensor.h"
#include "band/interface/tensor_view.h"

namespace Band {

class ErrorReporter;
class Tensor;

class TensorRingBuffer {
 public:
  TensorRingBuffer(ErrorReporter* error_reporter,
                   std::vector<std::shared_ptr<Interface::ITensor>> tensors,
                   std::vector<int> tensor_indices, int size = 128);
  ~TensorRingBuffer();

  const int GetTensorsLength() const;
  int Alloc();
  bool IsTensorIndexValid(int tensor_index) const;
  bool IsHandleValid(int handle) const;
  BandStatus GetTensorFromHandle(Interface::ITensor* dst, int tensor_index,
                                 int handle) const;
  BandStatus PutTensorToHandle(const Interface::ITensor* src, int tensor_index,
                               int handle);
  BandStatus GetTensorsFromHandle(std::vector<Interface::ITensor*>& dst_tensors,
                                  int handle) const;
  BandStatus PutTensorsToHandle(
      const std::vector<Interface::ITensor*>& src_tensors, int handle);

 private:
  int GetIndex(int handle) const;
  BandStatus CopyTensors(const std::vector<Interface::ITensor*>& src_tensors,
                         std::vector<Interface::ITensor*>& dst_tensors) const;
  BandStatus CopyTensor(const Interface::ITensor* src,
                        Interface::ITensor* dst) const;

  mutable std::mutex head_mtx_;
  int head_ = 0;
  const int size_;
  std::vector<Interface::ITensor*>* tensors_;
  // Model's tensor index to ring buffer's index
  std::map<int, int> tensor_to_buffer_;
  ErrorReporter* error_reporter_;
};
}  // namespace Band

#endif  // BAND_TENSOR_RING_BUFFER_H_
