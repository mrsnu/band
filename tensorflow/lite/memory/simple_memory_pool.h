#ifndef TENSORFLOW_LITE_MEMORY_SIMPLE_MEMORY_POOL_H_
#define TENSORFLOW_LITE_MEMORY_SIMPLE_MEMORY_POOL_H_
// memcpy
#include <cstring>
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
template <typename T>
T AlignTo(size_t alignment, T offset) {
  return offset % alignment == 0 ? offset
                                 : offset + (alignment - offset % alignment);
}

class MemoryBlock {
 public:
  MemoryBlock() {}
  MemoryBlock(size_t size_bytes, char* buffer)
      : size_bytes_(size_bytes), buffer_(buffer) {}

  char* const GetBuffer() const { return buffer_; }
  char* const GetBuffer() { return buffer_; }
  size_t GetBufferSize() const { return size_bytes_; }

 protected:
  size_t size_bytes_;
  char* buffer_;
};

template <typename T>
class SimpleMemoryPool : public MemoryBlock {
 public:
  SimpleMemoryPool() {}
  // ctor for TensorMemoryPool
  SimpleMemoryPool(ErrorReporter* error_reporter, size_t size_bytes,
                   size_t alignment = kTfLiteTensorDefaultAlignment)
      : MemoryBlock(size_bytes, new char[size_bytes]),
        error_reporter_(error_reporter),
        alignment_(alignment),
        own_buffer_(true) {}
  SimpleMemoryPool(size_t size_bytes, char* buffer = nullptr,
                   size_t alignment = kTfLiteTensorDefaultAlignment)
      : MemoryBlock(size_bytes,
                    buffer == nullptr ? new char[size_bytes] : buffer),
        error_reporter_(nullptr),
        alignment_(alignment),
        own_buffer_(buffer == nullptr) {}

  size_t GetHead() const { return head_; }
  size_t GetOffset(int handle) const { return handle_offsets_.at(handle).GetBuffer() - buffer_; }
  char* const GetBufferFromHandle(int handle) {
    if (handle_offsets_.find(handle) == handle_offsets_.end()) {
      return nullptr;
    } else {
      return handle_offsets_[handle].GetBuffer();
    }
  }

  virtual ~SimpleMemoryPool() {
    if (own_buffer_) {
      delete[] buffer_;
    }
  }

  virtual TfLiteStatus Allocate(size_t size_bytes, int handle) {
    auto current_block_it = handle_offsets_.find(handle);
    TF_LITE_ENSURE_EQ(error_reporter_, current_block_it, handle_offsets_.end());

    size_bytes = AlignTo(alignment_, size_bytes);
    // search for an empty block that perfectly fits
    for (auto it = empty_blocks_.begin(); it != empty_blocks_.end(); ++it) {
      if (size_bytes == it->second) {
        handle_offsets_[handle] = T(it->second, buffer_ + it->first);
        empty_blocks_.erase(it);
        return kTfLiteOk;
      }
    }

    // resize if needed
    if (size_bytes + head_ > size_bytes_) {
      if (Resize(size_bytes_ * 2) != kTfLiteOk) {
        return kTfLiteError;
      } else {
        TF_LITE_MAYBE_KERNEL_LOG(
            error_reporter_,
            "Allocate: Resize to %d might need to use more initial memory.",
            size_bytes);
      }
    }

    // allocate new block and move head
    handle_offsets_[handle] = T(size_bytes, buffer_ + head_);
    head_ += size_bytes;
    return kTfLiteOk;
  }

  virtual TfLiteStatus Deallocate(int handle) {
    auto current_block_it = handle_offsets_.find(handle);
    TF_LITE_ENSURE_NEQ(error_reporter_, current_block_it, handle_offsets_.end());

    T current_block = handle_offsets_[handle];
    handle_offsets_.erase(current_block_it);

    // first add to empty block
    empty_blocks_[current_block.GetBuffer() - buffer_] =
        current_block.GetBufferSize();

    // move head forward if possible
    auto it = empty_blocks_.rbegin();
    while (it != empty_blocks_.rend() && it->first + it->second == head_) {
      head_ = it->first;
      it = decltype(it){empty_blocks_.erase(std::next(it).base())};
    }

    return kTfLiteOk;
  }

  virtual TfLiteStatus Resize(size_t size_bytes) {
    TF_LITE_ENSURE_EQ(error_reporter_, size_bytes % alignment_, 0);
    TF_LITE_ENSURE(error_reporter_, own_buffer_);
    char* buffer = new char[size_bytes];
    if (buffer_) {
      memcpy(buffer, buffer_, head_);
      delete[] buffer_;
    }
    buffer_ = buffer;
    size_bytes_ = size_bytes;
    return kTfLiteOk;
  }

 protected:
  ErrorReporter* error_reporter_;
  size_t head_ = 0;
  // map of (handle, allocated memory block)
  std::map<int, T> handle_offsets_;
  // map of (offset, size of empty block)
  // allocate only returns `perfectly matched`
  // blocks to avoid fragmentation of memory
  // based on the assumption that every amount of
  // memory request for same model will always be identical
  std::map<size_t, size_t> empty_blocks_;

  size_t alignment_;
  bool own_buffer_;
};
}  // namespace tflite

#endif TENSORFLOW_LITE_MEMORY_SIMPLE_MEMORY_POOL_H_
