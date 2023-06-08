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
  virtual bool IsCompatible(const Buffer* input) const = 0;
  void SetOutput(std::shared_ptr<Buffer> output) { output_ = output; }
  std::shared_ptr<Buffer> GetOutput() const { return output_; }

 private:
  std::shared_ptr<Buffer> output_;
};

class CropOperation : public IOperation {
 public:
  CropOperation(int x0, int y0, int x1, int y1)
      : x0_(x0), y0_(y0), x1_(x1), y1_(y1) {}
  absl::Status Process(const Buffer* input) override;
  bool IsCompatible(const Buffer* input) const override;

 private:
  int x0_, y0_, x1_, y1_;
};

class ResizeOperation : public IOperation {
 public:
  ResizeOperation(const std::vector<int>& dims) : dims_(dims) {}
  absl::Status Process(const Buffer* input) override;
  bool IsCompatible(const Buffer* input) const override;

 private:
  std::vector<int> dims_;
};

class ConvertOperation : public IOperation {
 public:
  ConvertOperation(FormatType format_type) : format_type_(format_type) {}
  absl::Status Process(const Buffer* input) override;
  bool IsCompatible(const Buffer* input) const override;

 private:
  FormatType format_type_;
};

}  // namespace tensor
}  // namespace band

#endif