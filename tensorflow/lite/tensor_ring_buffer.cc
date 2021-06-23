#include <cstring>  // memcpy

#include "tensorflow/lite/tensor_ring_buffer.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
TensorRingBuffer::TensorRingBuffer(ErrorReporter* error_reporter,
                                   std::vector<const TfLiteTensor*> tensors,
                                   size_t size)
    : error_reporter_(error_reporter),
      tensors_(new std::vector<TfLiteTensor>[size]),
      size_(size) {
  for (size_t i = 0; i < size_; i++) {
    tensors_[i].resize(tensors.size());
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      TfLiteIntArray* shape = TfLiteIntArrayCopy(tensors[j]->dims);
      TfLiteTensorReset(tensors[j]->type, tensors[j]->name, shape,
                        tensors[j]->params, nullptr, 0, kTfLiteDynamic, nullptr,
                        tensors[j]->is_variable, &tensors_[i][j]);
      TfLiteTensorRealloc(tensors[j]->bytes, &tensors_[i][j]);
    }
  }
}

TensorRingBuffer::~TensorRingBuffer() {
  for (size_t i = 0; i < size_; i++) {
    for (size_t j = 0; j < tensors_[i].size(); j++) {
      TfLiteTensorFree(&tensors_[i][j]);
    }
  }

  delete[] tensors_;
}

int TensorRingBuffer::Alloc() { return head_++; }

bool TensorRingBuffer::IsValid(int handle) const {
  if (handle >= 0 && (head_ / size_) > (handle / size_) &&
      GetIndex(head_) >= handle) {
    return false;
  } else {
    return true;
  }
}

const std::vector<TfLiteTensor>* TensorRingBuffer::Get(int handle) const {
  if (IsValid(handle)) {
    return &tensors_[GetIndex(handle)];
  } else {
    TF_LITE_REPORT_ERROR(error_reporter_, "Invalid memory handle: %d head: %d.", handle, head_);
    return nullptr;
  }
}

TfLiteStatus TensorRingBuffer::Put(std::vector<TfLiteTensor> tensors,
                                   int handle) {
  if (!IsValid(handle)) {
    TF_LITE_REPORT_ERROR(error_reporter_, "Invalid memory handle: %d head: %d.", handle, head_);
    return kTfLiteError;
  }

  size_t index = GetIndex(handle);

  if (tensors.size() != tensors_[index].size()) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Invalid tensor size: %d expected: %d", tensors.size(),
                         tensors_[index].size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < tensors.size(); i++) {
    TfLiteTensor& src = tensors[i];
    TfLiteTensor* dst = &tensors_[index][i];

    if (!TfLiteIntArrayEqual(src.dims, dst->dims)) {
      TF_LITE_REPORT_ERROR(error_reporter_,
                           "Tensor assignment to different size.");
      return kTfLiteError;
    }

    std::memcpy(dst->data.raw, src.data.raw, dst->bytes);
  }

  return kTfLiteOk;
}

size_t TensorRingBuffer::GetIndex(int handle) const { return handle % size_; }
}  // namespace tflite