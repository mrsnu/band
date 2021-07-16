#include <iostream>
#include "tensorflow/lite/red_black_tree.h"

namespace tflite {

void RedBlackTree::Init(Node* nil) {
  nil_ = nil;
  nil_->SetColor(NodeColor::BLACK);
  nil_->left_ = nil_;
  nil_->right_ = nil_;
  nil_->value_ = 0;
  root_ = nil_;
}

RedBlackTree::Node* RedBlackTree::Search(const std::size_t v) {
  Node* x = root_;
  while (x != nil_) {
    if (v == x->value_)
      break;
    else if (v < x->value_)
      x = x->left_;
    else
      x = x->right_;
  }
  return x;
}

RedBlackTree::Node* RedBlackTree::SearchBest(const std::size_t v) {
  Node* y = nullptr;
  Node* x = root_;
  while (x != nil_) {
    y = x;
    if (v == x->value_)
      break;
    else if (v < x->value_)
      x = x->left_;
    else
      x = x->right_;
  }
  while (y != nullptr && v > y->value_) y = y->GetParent();
  return y;
}

RedBlackTree::Node* RedBlackTree::SearchAtLeast(const std::size_t v) {
  Node* x = root_;
  while (x != nil_) {
    if (v <= x->value_)
      return x;
    else
      x = x->right_;
  }
  return nullptr;
}

void RedBlackTree::Insert(Node* z) {
  Node* y = nullptr;
  Node* x = root_;
  while (x != nil_) {
    y = x;
    if (z->value_ < x->value_)
      x = x->left_;
    else
      x = x->right_;
  }

  z->SetParent(y);
  if (y == nullptr)
    root_ = z;
  else if (z->value_ < y->value_)
    y->left_ = z;
  else
    y->right_ = z;

  z->left_ = nil_;
  z->right_ = nil_;
  z->SetColor(NodeColor::RED);

  InsertFixup(z);
}

void RedBlackTree::InsertFixup(Node* z) {
  while (z != root_ && z->GetParent()->GetColor() == NodeColor::RED) {
    if (z->GetParent() == z->GetParent()->GetParent()->left_) {
      Node* y = z->GetParent()->GetParent()->right_;
      if (y->GetColor() == NodeColor::RED) {
        z->GetParent()->SetColor(NodeColor::BLACK);
        y->SetColor(NodeColor::BLACK);
        z->GetParent()->GetParent()->SetColor(NodeColor::RED);
        z = z->GetParent()->GetParent();
      } else {
        if (z == z->GetParent()->right_) {
          z = z->GetParent();
          LeftRotate(z);
        }
        z->GetParent()->SetColor(NodeColor::BLACK);
        z->GetParent()->GetParent()->SetColor(NodeColor::RED);
        RightRotate(z->GetParent()->GetParent());
      }
    } else {
      Node* y = z->GetParent()->GetParent()->left_;
      if (y->GetColor() == NodeColor::RED) {
        z->GetParent()->SetColor(NodeColor::BLACK);
        y->SetColor(NodeColor::BLACK);
        z->GetParent()->GetParent()->SetColor(NodeColor::RED);
        z = z->GetParent()->GetParent();
      } else {
        if (z == z->GetParent()->left_) {
          z = z->GetParent();
          RightRotate(z);
        }
        z->GetParent()->SetColor(NodeColor::BLACK);
        z->GetParent()->GetParent()->SetColor(NodeColor::RED);
        LeftRotate(z->GetParent()->GetParent());
      }
    }
  }
  root_->SetColor(NodeColor::BLACK);
  root_->prev_size_ = 0;
}

void RedBlackTree::Remove(Node* z) {
  Node* x = nullptr;
  Node* y = z;
  NodeColor yOriginalColor = y->GetColor();
  if (z->left_ == nil_) {
    x = z->right_;
    Transplant(z, z->right_);
  } else if (z->right_ == nil_) {
    x = z->left_;
    Transplant(z, z->left_);
  } else {
    y = Successor(z);
    yOriginalColor = y->GetColor();
    x = y->right_;
    if (y->GetParent() == z) {
      x->SetParent(y);
    } else {
      Transplant(y, y->right_);
      y->right_ = z->right_;
      if (y->right_ != nil_) y->right_->SetParent(y);
    }
    Transplant(z, y);
    y->left_ = z->left_;
    if (y->left_ != nil_) y->left_->SetParent(y);
    y->SetColor(z->GetColor());
  }

  if (yOriginalColor == NodeColor::BLACK) RemoveFixup(x);
}

void RedBlackTree::RemoveFixup(Node* z) {
  while (z != root_ && z->GetColor() == NodeColor::BLACK) {
    Node* w = nullptr;
    if (z->GetParent()->left_ == z) {
      w = z->GetParent()->right_;
      if (w->GetColor() == NodeColor::RED) {
        w->SetColor(NodeColor::BLACK);
        z->GetParent()->SetColor(NodeColor::RED);
        LeftRotate(z->GetParent());
        w = z->GetParent()->right_;
      }
      if ((w->right_ == nil_ || w->right_->GetColor() == NodeColor::BLACK) &&
          (w->left_ == nil_ || w->left_->GetColor() == NodeColor::BLACK)) {
        w->SetColor(NodeColor::RED);
        z = z->GetParent();
      } else {
        if (w->right_ == nil_ || w->right_->GetColor() == NodeColor::BLACK) {
          w->left_->SetColor(NodeColor::BLACK);
          w->SetColor(NodeColor::RED);
          RightRotate(w);
          w = z->GetParent()->right_;
        }
        w->SetColor(z->GetParent()->GetColor());
        z->GetParent()->SetColor(NodeColor::BLACK);
        w->right_->SetColor(NodeColor::BLACK);
        LeftRotate(z->GetParent());
        z = root_;
      }
    } else {
      w = z->GetParent()->left_;
      if (w->GetColor() == NodeColor::RED) {
        w->SetColor(NodeColor::BLACK);
        z->GetParent()->SetColor(NodeColor::RED);
        RightRotate(z->GetParent());
        w = z->GetParent()->left_;
      }
      if ((w->right_ == nil_ || w->right_->GetColor() == NodeColor::BLACK) &&
          (w->left_ == nil_ || w->left_->GetColor() == NodeColor::BLACK)) {
        w->SetColor(NodeColor::RED);
        z = z->GetParent();
      } else {
        if (w->left_ == nil_ || w->left_->GetColor() == NodeColor::BLACK) {
          w->right_->SetColor(NodeColor::BLACK);
          w->SetColor(NodeColor::RED);
          LeftRotate(w);
          w = z->GetParent()->left_;
        }
        w->SetColor(z->GetParent()->GetColor());
        z->GetParent()->SetColor(NodeColor::BLACK);
        w->left_->SetColor(NodeColor::BLACK);
        RightRotate(z->GetParent());
        z = root_;
      }
    }
  }
  z->SetColor(NodeColor::BLACK);
}

RedBlackTree::Node* RedBlackTree::Successor(Node* x) {
  x = x->right_;
  while (x->left_ != nil_) {
    x = x->left_;
  }
  return x;
}

void RedBlackTree::Print() const {
  if (root_->right_ != nil_) {
    Print(root_->right_, true, "");
  }
  std::cout << root_->value_ << '\n';
  if (root_->left_ != nil_) {
    Print(root_->left_, false, "");
  }
}

void RedBlackTree::Print(Node* x, bool isRight, std::string indent) const {
  if (x->right_ != nil_) {
    Print(x->right_, true, indent + (isRight ? "        " : " |      "));
  }
  std::cout << indent;
  if (isRight) {
    std::cout << " /";
  } else {
    std::cout << " \\";
  }
  std::cout << "----- ";
  std::cout << x->value_ << '\n';
  if (x->left_ != nil_) {
    Print(x->left_, false, indent + (isRight ? " |      " : "        "));
  }
}

void RedBlackTree::Transplant(Node* u, Node* v) {
  Node* uParent = u->GetParent();
  if (uParent == nullptr)
    root_ = v;
  else if (u == uParent->left_)
    uParent->left_ = v;
  else
    uParent->right_ = v;
  v->SetParent(uParent);
}

void RedBlackTree::LeftRotate(Node* x) {
  Node* xParent = x->GetParent();
  Node* y = x->right_;
  x->right_ = y->left_;
  if (y->left_ != nil_) y->left_->SetParent(x);
  y->SetParent(xParent);
  if (xParent == nullptr)
    root_ = y;
  else if (xParent->left_ == x)
    xParent->left_ = y;
  else
    xParent->right_ = y;
  y->left_ = x;
  x->SetParent(y);
}

void RedBlackTree::RightRotate(Node* x) {
  Node* xParent = x->GetParent();
  Node* y = x->left_;
  x->left_ = y->right_;
  if (y->right_ != nil_) y->right_->SetParent(x);
  y->SetParent(xParent);
  if (xParent == nullptr)
    root_ = y;
  else if (xParent->left_ == x)
    xParent->left_ = y;
  else
    xParent->right_ = y;
  y->right_ = x;
  x->SetParent(y);
}
}  // namespace tflite