#include <cassert>
#include <cstring>  // memcpy
#include <mutex>

#include "tensorflow/lite/tensor_ring_buffer.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
TensorRingBuffer::TensorRingBuffer(ErrorReporter* error_reporter,
                                   Tensors tensors,
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

bool TensorRingBuffer::IsValid(int handle) const {
  return (handle >= 0) && (head_ - size_ <= handle) && (handle < head_);
}

TfLiteStatus TensorRingBuffer::GetTensorsFromHandle(Tensors& dst_tensors, int handle, std::vector<bool> mask) const {
  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "GetTensorsFromHandle: Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }
  return CopyTensors(tensors_[GetIndex(handle)], dst_tensors, mask);
}

TfLiteStatus TensorRingBuffer::PutTensorsToHandle(const Tensors& src_tensors,
                                   int handle, std::vector<bool> mask) {
  std::lock_guard<std::mutex> lock(head_mtx_);
  if (!IsValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "PutTensorsToHandle: Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  return CopyTensors(src_tensors, tensors_[GetIndex(handle)], mask);
}

TfLiteStatus TensorRingBuffer::CopyTensors(const Tensors& src_tensors,
                                           Tensors& dst_tensors, std::vector<bool> mask) const {
  const int tensors_length = GetTensorsLength();
  if ((src_tensors.size() != tensors_length) ||
      (dst_tensors.size() != tensors_length)) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Invalid tensor length. src tensors: %d dst tensors: %d expected: %d",
        src_tensors.size(), dst_tensors.size(), tensors_length);
    return kTfLiteError;
  }

  if (mask.size() == 0) {
    mask = std::vector<bool>(true, tensors_length);
  }

  if (mask.size() != tensors_length) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Invalid mask length. mask: %d expected: %d",
        mask.size(), tensors_length);
    return kTfLiteError;
  }

  for (size_t i = 0; i < tensors_length; i++) {
    if (!mask[i]) {
      continue;
    }
    
    const TfLiteTensor* src = src_tensors[i];
    TfLiteTensor* dst = dst_tensors[i];

    if (TfLiteTensorDataCopy(src, dst) == kTfLiteError) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Tensor data copy failure. src name : %s, dst name : %s", src->name,
          dst->name);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

int TensorRingBuffer::GetIndex(int handle) const { return handle % size_; }
}  // namespace tflite
