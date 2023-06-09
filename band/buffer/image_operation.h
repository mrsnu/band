#ifndef BAND_BUFFER_IMAGE_OPERATION_H
#define BAND_BUFFER_IMAGE_OPERATION_H
#include "absl/status/status.h"
#include "band/buffer/buffer.h"
#include "band/buffer/operation.h"

namespace band {

class CropOperation : public IOperation {
 public:
  CropOperation(int x0, int y0, int x1, int y1)
      : x0_(x0), y0_(y0), x1_(x1), y1_(y1) {}

  virtual IOperation* Clone() const override {
    return new CropOperation(*this);
  }
  virtual absl::Status Process(const Buffer& input) override;
  virtual absl::Status IsValid(const Buffer& input) const override;

 private:
  int x0_, y0_, x1_, y1_;
};

class ResizeOperation : public IOperation {
 public:
  ResizeOperation(size_t width, size_t height) : dims_({width, height}) {}
  ResizeOperation(const std::vector<size_t>& dims) : dims_(dims) {}

  virtual IOperation* Clone() const override {
    return new ResizeOperation(*this);
  }
  virtual absl::Status Process(const Buffer& input) override;
  virtual absl::Status IsValid(const Buffer& input) const override;

 private:
  std::vector<size_t> dims_;
};

// RotateOperation rotates the input buffer by the specified angle in degrees
// (counter-clockwise). The output buffer will be created automatically.
class RotateOperation : public IOperation {
 public:
  RotateOperation(int angle_deg) : angle_deg_(angle_deg) {}

  virtual IOperation* Clone() const override {
    return new RotateOperation(*this);
  }
  virtual absl::Status Process(const Buffer& input) override;
  virtual absl::Status IsValid(const Buffer& input) const override;

 private:
  int angle_deg_;
};

class FlipOperation : public IOperation {
 public:
  // horizontal: true for horizontal flip, false for vertical flip
  FlipOperation(bool horizontal) : horizontal_(horizontal) {}

  virtual IOperation* Clone() const override {
    return new FlipOperation(*this);
  }
  virtual absl::Status Process(const Buffer& input) override;
  virtual absl::Status IsValid(const Buffer& input) const override;

 private:
  bool horizontal_;
};

class ConvertOperation : public IOperation {
 public:
  ConvertOperation() = default;

  virtual IOperation* Clone() const override {
    return new ConvertOperation(*this);
  }
  virtual absl::Status Process(const Buffer& input) override;
  virtual absl::Status IsValid(const Buffer& input) const override;
};
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_OPERATION_H