#include <cstring>  // memcpy

#include "tensorflow/lite/tensor_ring_buffer.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
TensorRingBuffer::TensorRingBuffer(ErrorReporter* error_reporter,
                                   Tensors tensors,
                                   int size)
    : error_reporter_(error_reporter),
      tensors_(new std::vector<TfLiteTensor*>[size]),
      size_(size) {
  for (size_t i = 0; i < size_; i++) {
    tensors_[i].resize(tensors.size());
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      tensors_[i][j] = TfLiteTensorCopy(tensors[j]);
    }
  }
}

TensorRingBuffer::~TensorRingBuffer() {
  for (size_t i = 0; i < size_; i++) {
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      TfLiteTensorFree(tensors_[i][j]);
      free(tensors_[i][j]);
    }
  }

  delete[] tensors_;
}

int TensorRingBuffer::Alloc() { return head_++; }

bool TensorRingBuffer::IsValid(int handle) const {
  return (handle >= 0) && (head_ - size_ <= handle) && (handle < head_);
}

TfLiteStatus TensorRingBuffer::Get(Tensors& dst_tensors, int handle) const {
  if (!IsValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  return CopyTensors(tensors_[GetIndex(handle)], dst_tensors);
}

TfLiteStatus TensorRingBuffer::Put(const Tensors& src_tensors,
                                   int handle) {
  if (!IsValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  return CopyTensors(src_tensors, tensors_[GetIndex(handle)]);
}

TfLiteStatus TensorRingBuffer::CopyTensors(const Tensors& src_tensors, Tensors& dst_tensors) const {
  if (src_tensors.size() != dst_tensors.size()) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Invalid tensor size: %d expected: %d", src_tensors.size(),
                         dst_tensors.size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < src_tensors.size(); i++) {
    const TfLiteTensor* src = src_tensors[i];
    TfLiteTensor* dst = dst_tensors[i];

    if (TfLiteTensorDataCopy(src, dst) == kTfLiteError) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Tensor data copy failure. src name : %s, dst name : %s", src->name, dst->name);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

int TensorRingBuffer::GetIndex(int handle) const { return handle % size_; }
}  // namespace tflite
