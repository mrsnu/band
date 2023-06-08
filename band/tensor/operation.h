#ifndef BAND_TENSOR_OPERATION_H
#define BAND_TENSOR_OPERATION_H

#include "absl/status/status.h"
#include "band/tensor/buffer.h"

namespace band {
namespace tensor {

class IOperation {
 public:
  virtual ~IOperation() = default;
  virtual absl::Status Process(const Buffer* input) = 0;
  virtual absl::Status IsValid(const Buffer* input) const = 0;
  // explicitly assign output buffer, otherwise it will be created automatically
  void SetOutput(std::shared_ptr<Buffer> output) { output_ = output; }
  std::shared_ptr<Buffer> GetOutput() const { return output_; }

 protected:
  std::shared_ptr<Buffer> output_;
};

class CropOperation : public IOperation {
 public:
  CropOperation(int x0, int y0, int x1, int y1)
      : x0_(x0), y0_(y0), x1_(x1), y1_(y1) {}
  absl::Status Process(const Buffer* input) override;
  absl::Status IsValid(const Buffer* input) const override;

 private:
  int x0_, y0_, x1_, y1_;
};

class ResizeOperation : public IOperation {
 public:
  ResizeOperation(const std::vector<int>& dims) : dims_(dims) {}
  absl::Status Process(const Buffer* input) override;
  absl::Status IsValid(const Buffer* input) const override;

 private:
  std::vector<int> dims_;
};

class RotateOperation : public IOperation {
 public:
  RotateOperation(int angle_deg) : angle_deg_(angle_deg) {}
  absl::Status Process(const Buffer* input) override;
  absl::Status IsValid(const Buffer* input) const override;

 private:
  int angle_deg_;
};

class FlipOperation : public IOperation {
 public:
  // horizontal: true for horizontal flip, false for vertical flip
  FlipOperation(bool horizontal) : horizontal_(horizontal) {}
  absl::Status Process(const Buffer* input) override;
  absl::Status IsValid(const Buffer* input) const override;

 private:
  bool horizontal_;
};

class ConvertOperation : public IOperation {
 public:
  ConvertOperation(FormatType format_type) : format_type_(format_type) {}
  absl::Status Process(const Buffer* input) override;
  absl::Status IsValid(const Buffer* input) const override;

 private:
  FormatType format_type_;
};

}  // namespace tensor
}  // namespace band

#endif