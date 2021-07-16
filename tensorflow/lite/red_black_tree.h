#ifndef TENSORFLOW_LITE_RBTREE_H
#define TENSORFLOW_LITE_RBTREE_H

#include <string>

namespace tflite {
class RedBlackTree {
 public:
  enum class NodeColor : std::size_t { BLACK, RED };
  class Node {
   public:
    std::size_t prev_size_;

   private:
    Node* parent_;

   public:
    Node* right_;
    Node* left_;
    std::size_t value_;

    inline Node* GetParentRaw() { return parent_; }

    inline Node* GetParent() {
      return reinterpret_cast<Node*>((std::size_t)parent_ >> 2 << 2);
    }

    inline void SetParent(Node* p) {
      parent_ = reinterpret_cast<Node*>((std::size_t)p |
                                        ((std::size_t)parent_ & 2) | 1);
    }

    inline NodeColor GetColor() {
      if (parent_ == nullptr) return NodeColor::BLACK;
      return (std::size_t)parent_ & 2 ? NodeColor::RED : NodeColor::BLACK;
    }

    inline void SetColor(NodeColor color) {
      if (color == NodeColor::RED)
        parent_ = reinterpret_cast<Node*>((std::size_t)parent_ | 2);
      else
        parent_ =
            reinterpret_cast<Node*>((std::size_t)parent_ & ~((std::size_t)2));
    }
  };

 private:
  Node* nil_;
  Node* root_;

 public:
  void Init(Node* nil);
  Node* Search(const std::size_t v);
  Node* SearchBest(const std::size_t v);
  Node* SearchAtLeast(const std::size_t v);
  void Insert(Node* z);
  void Remove(Node* z);
  Node* Successor(Node* x);
  void Print() const;

 private:
  void InsertFixup(Node* z);
  void RemoveFixup(Node* z);
  void Transplant(Node* u, Node* v);
  void LeftRotate(Node* x);
  void RightRotate(Node* x);
  void Print(Node* x, bool isRight, std::string indent) const;
};
};  // namespace tflite

#endif  // TENSORFLOW_LITE_RBTREE_H