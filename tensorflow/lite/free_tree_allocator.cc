#include "tensorflow/lite/free_tree_allocator.h"
#include "tensorflow/lite/red_black_tree.h"

#include <algorithm>
#include <cassert>
#include <cstddef> // max_align_t
#include <memory>
#include <string>

#include <iostream>

namespace tflite {
FreeTreeAllocator::FreeTreeAllocator(const std::size_t size) : size_(size) {
  static std::size_t rootNodePadding = GetRootNodePadding();
  static std::string message =
      "Total size must be at least " +
      std::to_string(sizeof(RedBlackTree::Node) * 2 + rootNodePadding) +
      " bytes for an allocator with at least " +
      std::to_string(sizeof(RedBlackTree::Node) - sizeof(Header)) +
      " bytes of free space";
  assert(size >= sizeof(RedBlackTree::Node) * 2 + rootNodePadding && message.c_str());
  start_address_ = ::operator new(size);
  Reset();
}

FreeTreeAllocator::~FreeTreeAllocator() {
  ::operator delete(start_address_);
  start_address_ = nullptr;
}

void* FreeTreeAllocator::Allocate(const std::size_t size) {
  std::size_t padding = size + sizeof(Header) < sizeof(RedBlackTree::Node)
                            ? sizeof(RedBlackTree::Node) - sizeof(Header) - size
                            : 0;
  void* currentAddress = (void*)(sizeof(Header) + size + padding);
  void* nextAddress = (void*)(sizeof(Header) + size + padding);
  std::size_t space = size + padding + sizeof(std::max_align_t);
  std::align(alignof(std::max_align_t), sizeof(std::max_align_t), nextAddress,
             space);
  padding += (std::size_t)nextAddress - (std::size_t)currentAddress;
  
  std::unique_lock<std::mutex> tree_lock(tree_mtx_);
  RedBlackTree::Node* node = tree_.SearchBest(size + padding);

  if (node == nullptr) {
    return nullptr;
  }

  tree_.Remove(node);

  if (node->value_ >= size + padding + sizeof(RedBlackTree::Node)) {
    RedBlackTree::Node* splittedNode = reinterpret_cast<RedBlackTree::Node*>(
        reinterpret_cast<char*>(node) + sizeof(Header) + size + padding);
    splittedNode->value_ = node->value_ - (size + padding + sizeof(Header));
    tree_.Insert(splittedNode);
    std::size_t* nextBlockAddress =
        reinterpret_cast<std::size_t*>(reinterpret_cast<char*>(splittedNode) +
                                       sizeof(Header) + splittedNode->value_);
    if ((std::size_t)nextBlockAddress <=
        (std::size_t)start_address_ + size_ - sizeof(std::size_t)) {
      *nextBlockAddress = sizeof(Header) + splittedNode->value_;
    }
  } else {
    padding += node->value_ - (size + padding);
  }
  tree_lock.unlock();

  Header* header = reinterpret_cast<Header*>(node);
  header->size_ = size + padding;

  *reinterpret_cast<std::size_t*>(reinterpret_cast<char*>(header) +
                                  sizeof(Header) + header->size_) = 0;

  return reinterpret_cast<char*>(node) + sizeof(Header);
}

void FreeTreeAllocator::Deallocate(void* ptr) {
  Header* header =
      reinterpret_cast<Header*>(reinterpret_cast<char*>(ptr) - sizeof(Header));
  RedBlackTree::Node* node = reinterpret_cast<RedBlackTree::Node*>(header);
  node->value_ = header->size_;
  Coalescence(node);
}

void FreeTreeAllocator::Reset() {
  std::lock_guard<std::mutex> tree_lock(tree_mtx_);
  RedBlackTree::Node* nil = reinterpret_cast<RedBlackTree::Node*>(start_address_);
  tree_.Init(nil);
  void* currentAddress =
      reinterpret_cast<RedBlackTree::Node*>(reinterpret_cast<char*>(start_address_) +
                                      sizeof(RedBlackTree::Node) + sizeof(Header));
  std::size_t space = size_ - sizeof(Header) - sizeof(RedBlackTree::Node);
  std::align(alignof(std::max_align_t), sizeof(std::max_align_t),
             currentAddress, space);
  RedBlackTree::Node* root = reinterpret_cast<RedBlackTree::Node*>(
      reinterpret_cast<char*>(currentAddress) - sizeof(Header));
  root->value_ = reinterpret_cast<char*>(start_address_) + size_ -
                  reinterpret_cast<char*>(root) - sizeof(Header);
  tree_.Insert(root);
}

void FreeTreeAllocator::Coalescence(RedBlackTree::Node* curr) {
  std::lock_guard<std::mutex> tree_lock(tree_mtx_);
  RedBlackTree::Node* next = reinterpret_cast<RedBlackTree::Node*>(
      reinterpret_cast<char*>(curr) + sizeof(Header) + curr->value_);
  if (((std::size_t)next < (std::size_t)start_address_ + size_) &&
      (std::size_t)next->GetParentRaw() & 1) {
    curr->value_ += next->value_ + sizeof(Header);
    tree_.Remove(next);
  }

  if (curr->prev_size_ != 0) {
    RedBlackTree::Node* prev = reinterpret_cast<RedBlackTree::Node*>(
        reinterpret_cast<char*>(curr) - curr->prev_size_);
    tree_.Remove(prev);
    prev->value_ += curr->value_ + sizeof(Header);
    tree_.Insert(prev);
    std::size_t* nextBlockAddress = reinterpret_cast<std::size_t*>(
        reinterpret_cast<char*>(prev) + sizeof(Header) + prev->value_);
    if ((std::size_t)nextBlockAddress <=
        (std::size_t)start_address_ + size_ - sizeof(std::size_t)) {
      *nextBlockAddress = sizeof(Header) + prev->value_;
    }
  } else {
    tree_.Insert(curr);
    std::size_t* nextBlockAddress = reinterpret_cast<std::size_t*>(
        reinterpret_cast<char*>(curr) + sizeof(Header) + curr->value_);
    if ((std::size_t)nextBlockAddress <=
        (std::size_t)start_address_ + size_ - sizeof(std::size_t)) {
      *nextBlockAddress = sizeof(Header) + curr->value_;
    }
  }
}

std::size_t FreeTreeAllocator::GetRootNodePadding() {
  void* currentAddress =
      reinterpret_cast<RedBlackTree::Node*>(sizeof(RedBlackTree::Node) + sizeof(Header));
  void* nextAddress = currentAddress;
  std::size_t space =
      sizeof(RedBlackTree::Node) * 3 - sizeof(Header) - sizeof(RedBlackTree::Node);
  std::align(alignof(std::max_align_t), sizeof(std::max_align_t), nextAddress,
             space);
  return (std::size_t)nextAddress - (std::size_t)currentAddress;
}
}  // namespace tflite
