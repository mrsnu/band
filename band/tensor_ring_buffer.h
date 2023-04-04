#ifndef BAND_TENSOR_RING_BUFFER_H_
#define BAND_TENSOR_RING_BUFFER_H_

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "band/interface/tensor.h"
#include "band/interface/tensor_view.h"

#include "absl/status/status.h"

namespace band {

class ErrorReporter;
class Tensor;

class TensorRingBuffer {
 public:
  TensorRingBuffer(ErrorReporter* error_reporter,
                   std::vector<std::shared_ptr<interface::ITensor>> tensors,
                   std::vector<int> tensor_indices, int size = 128);
  ~TensorRingBuffer();

  const int GetTensorsLength() const;
  int Alloc();
  bool IsTensorIndexValid(int tensor_index) const;
  bool IsHandleValid(int handle) const;
  absl::Status GetTensorFromHandle(interface::ITensor* dst, int tensor_index,
                                 int handle) const;
  absl::Status PutTensorToHandle(const interface::ITensor* src, int tensor_index,
                               int handle);
  absl::Status GetTensorsFromHandle(std::vector<interface::ITensor*>& dst_tensors,
                                  int handle) const;
  absl::Status PutTensorsToHandle(
      const std::vector<interface::ITensor*>& src_tensors, int handle);

 private:
  int GetIndex(int handle) const;
  absl::Status CopyTensors(const std::vector<interface::ITensor*>& src_tensors,
                         std::vector<interface::ITensor*>& dst_tensors) const;
  absl::Status CopyTensor(const interface::ITensor* src,
                        interface::ITensor* dst) const;

  mutable std::mutex head_mtx_;
  int head_ = 0;
  const int size_;
  std::vector<interface::ITensor*>* tensors_;
  // Model's tensor index to ring buffer's index
  std::map<int, int> tensor_to_buffer_;
  ErrorReporter* error_reporter_;
};
}  // namespace band

#endif  // BAND_TENSOR_RING_BUFFER_H_
