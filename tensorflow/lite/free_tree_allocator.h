#ifndef TENSORFLOW_LITE_FREE_TREE_ALLOCATOR_H_
#define TENSORFLOW_LITE_FREE_TREE_ALLOCATOR_H_

#include <vector>
#include <mutex>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/red_black_tree.h"

namespace tflite {
  // RBTree based free tree allocator
  // Borrowed from https://github.com/Kashio/A5
  class FreeTreeAllocator {
   public:
    FreeTreeAllocator(const std::size_t size);
    ~FreeTreeAllocator();
    void* Allocate(const std::size_t size);
    void Deallocate(void* ptr);
    void Print();
    void Reset();
    void* base() const { return start_address_; }
    static std::size_t GetRootNodePadding();

   private:
    struct Header {
      std::size_t prev_size_;
      std::size_t size_;
    };
    RedBlackTree tree_;
    std::mutex tree_mtx_;

    void Coalescence(RedBlackTree::Node* curr);
    std::size_t size_;
    void* start_address_;
  };
}  // namespace tflite

#endif  // TENSORFLOW_LITE_FREE_TREE_ALLOCATOR_H_
