// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/tensor_ring_buffer.h"

#include <cassert>
#include <cstring>  // memcpy
#include <mutex>

#include "absl/strings/str_format.h"
#include "band/interface/tensor.h"
#include "band/interface/tensor_view.h"
#include "band/tensor.h"

namespace band {
TensorRingBuffer::TensorRingBuffer(
    std::vector<std::shared_ptr<interface::ITensor>> tensors,
    std::vector<int> tensor_indices, int size)
    : tensors_(new std::vector<interface::ITensor*>[size]), size_(size) {
  assert(size_ > 0);
  for (size_t i = 0; i < size_; i++) {
    tensors_[i].resize(tensors.size());
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      tensors_[i][j] = new Tensor(tensors[j].get());
    }
  }

  for (int i = 0; i < tensor_indices.size(); i++) {
    tensor_to_buffer_[tensor_indices[i]] = i;
  }
}

TensorRingBuffer::~TensorRingBuffer() {
  for (size_t i = 0; i < size_; i++) {
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      delete tensors_[i][j];
    }
  }

  delete[] tensors_;
}

const int TensorRingBuffer::GetTensorsLength() const {
  return tensors_[0].size();
}

int TensorRingBuffer::Alloc() {
  std::lock_guard<std::mutex> lock(head_mtx_);
  return head_++;
}

bool TensorRingBuffer::IsTensorIndexValid(int tensor_index) const {
  return tensor_to_buffer_.find(tensor_index) != tensor_to_buffer_.end();
}

bool TensorRingBuffer::IsHandleValid(int handle) const {
  return (handle >= 0) && (head_ - size_ <= handle) && (handle < head_);
}

absl::Status TensorRingBuffer::GetTensorFromHandle(interface::ITensor* dst,
                                                   int tensor_index,
                                                   int handle) const {
  if (!IsTensorIndexValid(tensor_index)) {
    return absl::InternalError(absl::StrFormat(
        "GetTensorFromHandle: Invalid tensor index: %d.", tensor_index));
  }

  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    return absl::InternalError(absl::StrFormat(
        "GetTensorFromHandle: Invalid memory handle: %d head: %d.", handle,
        head_));
  }

  return CopyTensor(
      tensors_[GetIndex(handle)][tensor_to_buffer_.at(tensor_index)], dst);
}

absl::Status TensorRingBuffer::PutTensorToHandle(const interface::ITensor* src,
                                                 int tensor_index, int handle) {
  if (!IsTensorIndexValid(tensor_index)) {
    return absl::InternalError(absl::StrFormat(
        "PutTensorToHandle: Invalid tensor index: %d.", tensor_index));
  }

  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    return absl::InternalError(absl::StrFormat(
        "PutTensorToHandle: Invalid memory handle: %d head: %d.", handle,
        head_));
  }

  return CopyTensor(
      src, tensors_[GetIndex(handle)][tensor_to_buffer_.at(tensor_index)]);
}

absl::Status TensorRingBuffer::GetTensorsFromHandle(
    std::vector<interface::ITensor*>& dst_tensors, int handle) const {
  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    return absl::InternalError(absl::StrFormat(
        "GetTensorsFromHandle: Invalid memory handle: %d head: %d.", handle,
        head_));
  }
  return CopyTensors(tensors_[GetIndex(handle)], dst_tensors);
}

absl::Status TensorRingBuffer::PutTensorsToHandle(
    const std::vector<interface::ITensor*>& src_tensors, int handle) {
  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    return absl::InternalError(absl::StrFormat(
        "PutTensorsToHandle: Invalid memory handle: %d head: %d.", handle,
        head_));
  }

  return CopyTensors(src_tensors, tensors_[GetIndex(handle)]);
}

absl::Status TensorRingBuffer::CopyTensors(
    const std::vector<interface::ITensor*>& src_tensors,
    std::vector<interface::ITensor*>& dst_tensors) const {
  const int tensors_length = GetTensorsLength();
  if (src_tensors.size() != tensors_length ||
      dst_tensors.size() != tensors_length) {
    return absl::InternalError(absl::StrFormat(
        "Invalid tensor length. src tensors: %d dst tensors: %d expected: %d",
        src_tensors.size(), dst_tensors.size(), tensors_length));
  }

  for (size_t i = 0; i < tensors_length; i++) {
    if (!CopyTensor(src_tensors[i], dst_tensors[i]).ok()) {
      return absl::InternalError("Failed to copy tensors.");
    }
  }

  return absl::OkStatus();
}

absl::Status TensorRingBuffer::CopyTensor(const interface::ITensor* src,
                                          interface::ITensor* dst) const {
  if (!dst->CopyDataFrom(src).ok()) {
    return absl::InternalError(absl::StrFormat(
        "Tensor data copy failure. src name : %s, dst name : %s",
        src ? src->GetName() : "null", dst ? dst->GetName() : "null"));
  }
  return absl::OkStatus();
}

int TensorRingBuffer::GetIndex(int handle) const { return handle % size_; }
}  // namespace band
