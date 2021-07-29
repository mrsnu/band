#include <cassert>
#include <cstring>  // memcpy
#include <mutex>

#include "tensorflow/lite/tensor_ring_buffer.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
TensorRingBuffer::TensorRingBuffer(ErrorReporter* error_reporter,
                                   Tensors tensors,
                                   std::vector<int> tensor_indices,
                                   int size)
    : error_reporter_(error_reporter),
      tensors_(new std::vector<TfLiteTensor*>[size]),
      size_(size) {
  assert(size_ > 0);
  for (size_t i = 0; i < size_; i++) {
    tensors_[i].resize(tensors.size());
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      tensors_[i][j] = TfLiteTensorCreateLike(tensors[j]);
    }
  }

  for (int i = 0; i < tensor_indices.size(); i++) {
    model_to_buffer_[tensor_indices[i]] = i;
  }
}

TensorRingBuffer::~TensorRingBuffer() {
  for (size_t i = 0; i < size_; i++) {
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      TfLiteTensorDelete(tensors_[i][j]);
    }
  }

  delete[] tensors_;
}

const int TensorRingBuffer::GetTensorsLength() const { return tensors_[0].size(); }

int TensorRingBuffer::Alloc() {
  std::lock_guard<std::mutex> lock(head_mtx_);
  return head_++;
}

bool TensorRingBuffer::IsTensorIndexValid(int tensor_index) const {
  return model_to_buffer_.find(tensor_index) != model_to_buffer_.end();
}

bool TensorRingBuffer::IsHandleValid(int handle) const {
  return (handle >= 0) && (head_ - size_ <= handle) && (handle < head_);
}

TfLiteStatus TensorRingBuffer::GetTensorFromHandle(TfLiteTensor* dst,
                                                   int tensor_index,
                                                   int handle) const {
  if (!IsTensorIndexValid(tensor_index)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "GetTensorFromHandle: Invalid tensor index: %d.", tensor_index);
    return kTfLiteError;
  }

  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "GetTensorFromHandle: Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  return CopyTensor(tensors_[GetIndex(handle)][model_to_buffer_.at(tensor_index)],
                    dst);
}

TfLiteStatus TensorRingBuffer::PutTensorToHandle(const TfLiteTensor* src,
                                                 int tensor_index, int handle) {
  if (!IsTensorIndexValid(tensor_index)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "PutTensorToHandle: Invalid tensor index: %d.", tensor_index);
    return kTfLiteError;
  }

  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "PutTensorToHandle: Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  return CopyTensor(
      src, tensors_[GetIndex(handle)][model_to_buffer_.at(tensor_index)]);
}

TfLiteStatus TensorRingBuffer::GetTensorsFromHandle(Tensors& dst_tensors, int handle) const {
  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "GetTensorsFromHandle: Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }
  return CopyTensors(tensors_[GetIndex(handle)], dst_tensors);
}

TfLiteStatus TensorRingBuffer::PutTensorsToHandle(const Tensors& src_tensors,
                                   int handle) {
  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsHandleValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "PutTensorsToHandle: Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  return CopyTensors(src_tensors, tensors_[GetIndex(handle)]);
}

TfLiteStatus TensorRingBuffer::CopyTensors(const Tensors& src_tensors,
                                           Tensors& dst_tensors) const {
  const int tensors_length = GetTensorsLength();
  if (src_tensors.size() != tensors_length ||
      dst_tensors.size() != tensors_length) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Invalid tensor length. src tensors: %d dst tensors: %d expected: %d",
        src_tensors.size(), dst_tensors.size(), tensors_length);
    return kTfLiteError;
  }

  for (size_t i = 0; i < tensors_length; i++) {
    if (CopyTensor(src_tensors[i], dst_tensors[i]) != kTfLiteOk) {
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus TensorRingBuffer::CopyTensor(const TfLiteTensor* src,
                                          TfLiteTensor* dst) const {
  if (TfLiteTensorDataCopy(src, dst) == kTfLiteError) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Tensor data copy failure. src name : %s, dst name : %s", src->name,
        dst->name);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

int TensorRingBuffer::GetIndex(int handle) const { return handle % size_; }
}  // namespace tflite
