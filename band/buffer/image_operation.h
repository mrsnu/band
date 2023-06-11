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

  virtual IOperation* Clone() const override;
  virtual OperationType GetOperationType() const override;

 private:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

  int x0_, y0_, x1_, y1_;
};

class ResizeOperation : public IOperation {
 public:
  // width: -1 for auto, height: -1 for auto
  ResizeOperation(int width = -1, int height = -1) : dims_({width, height}) {}
  ResizeOperation(const std::vector<int>& dims) : dims_(dims) {}

  virtual IOperation* Clone() const override;
  virtual OperationType GetOperationType() const override;

 private:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;
  bool IsAuto(size_t dim) const { return dims_[dim] == -1; }

  std::vector<int> dims_;
};

// RotateOperation rotates the input buffer by the specified angle in degrees
// (counter-clockwise). The output buffer will be created automatically.
class RotateOperation : public IOperation {
 public:
  RotateOperation(int angle_deg) : angle_deg_(angle_deg) {}

  virtual IOperation* Clone() const override;
  virtual OperationType GetOperationType() const override;

 private:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

  int angle_deg_;
};

class FlipOperation : public IOperation {
 public:
  // horizontal: true for horizontal flip, false for vertical flip
  FlipOperation(bool horizontal) : horizontal_(horizontal) {}

  virtual IOperation* Clone() const override;
  virtual OperationType GetOperationType() const override;

 private:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

  bool horizontal_;
};

class ConvertOperation : public IOperation {
 public:
  ConvertOperation()
      : output_format_(BufferFormat::kRaw), is_format_specified_(false) {}
  ConvertOperation(BufferFormat buffer_format)
      : output_format_(buffer_format), is_format_specified_(true) {}

  virtual IOperation* Clone() const override;
  virtual OperationType GetOperationType() const override;

 private:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

  BufferFormat output_format_;
  bool is_format_specified_;
};
}  // namespace band

#endif  // BAND_BUFFER_IMAGE_OPERATION_H