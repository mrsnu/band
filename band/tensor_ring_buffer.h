/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_TENSOR_RING_BUFFER_H_
#define BAND_TENSOR_RING_BUFFER_H_

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "absl/status/status.h"
#include "band/interface/tensor.h"
#include "band/interface/tensor_view.h"

namespace band {

class Tensor;

class TensorRingBuffer {
 public:
  TensorRingBuffer(std::vector<std::shared_ptr<interface::ITensor>> tensors,
                   std::vector<int> tensor_indices, int size = 128);
  ~TensorRingBuffer();

  const int GetTensorsLength() const;
  int Alloc();
  bool IsTensorIndexValid(int tensor_index) const;
  bool IsHandleValid(int handle) const;
  absl::Status GetTensorFromHandle(interface::ITensor* dst, int tensor_index,
                                   int handle) const;
  absl::Status PutTensorToHandle(const interface::ITensor* src,
                                 int tensor_index, int handle);
  absl::Status GetTensorsFromHandle(
      std::vector<interface::ITensor*>& dst_tensors, int handle) const;
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
};
}  // namespace band

#endif  // BAND_TENSOR_RING_BUFFER_H_
