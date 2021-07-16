#ifndef TENSORFLOW_LITE_FREE_TREE_ALLOCATOR_H_
#define TENSORFLOW_LITE_FREE_TREE_ALLOCATOR_H_

#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/red_black_tree.h"

namespace tflite {
  class FreeTreeAllocator {
   public:
    FreeTreeAllocator(const std::size_t size);
    ~FreeTreeAllocator();
    void* Allocate(const std::size_t size, const std::size_t alignment);
    void Deallocate(void* ptr);
    void Reset();

   private:
    struct Header {
      std::size_t prev_size_;
      std::size_t size_;
    };
    RedBlackTree tree_;

    void Init();
    void Coalescence(RedBlackTree::Node* curr);
    static std::size_t GetRootNodePadding();
    std::size_t size_;
    void* start_address_;
  };
}  // namespace tflite

#endif  // TENSORFLOW_LITE_FREE_TREE_ALLOCATOR_H_
