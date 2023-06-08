#ifndef BAND_TENSOR_OPERATION_H
#define BAND_TENSOR_OPERATION_H

#include "absl/status/status.h"
#include "band/tensor/buffer.h"

namespace band {
namespace tensor {

// Interface for buffer operations such as crop, resize, rotate, flip, convert
// format, etc. Each operation should be able to validate an input buffer and
// process the input buffer to generate the output buffer.
// The output buffer can be explicitly assigned by calling SetOutput() or
// automatically created by the operation. Each operation should create
// output buffer if it is not explicitly assigned and cache the output buffer
// for future use (e.g. for the next operation with the same input format).
class IOperation {
 public:
  IOperation() = default;
  virtual ~IOperation() = default;
  IOperation(const IOperation&);
  // For processor builder to clone the operation
  virtual IOperation* Clone() const = 0;
  virtual absl::Status Process(const Buffer& input) = 0;
  virtual absl::Status IsValid(const Buffer& input) const = 0;
  // explicitly assign output buffer, otherwise it will be created automatically
  void SetOutput(Buffer* output);
  Buffer* GetOutput() const { return output_.get(); }

 protected:
  std::unique_ptr<Buffer> output_;
};
}  // namespace tensor
}  // namespace band

#endif