#ifndef BAND_BUFFER_OPERATION_H
#define BAND_BUFFER_OPERATION_H

#include "absl/status/status.h"
#include "band/buffer/buffer.h"

namespace band {
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
  virtual ~IOperation();
  // For processor builder to clone the operation
  virtual IOperation* Clone() const = 0;
  absl::Status Process(const Buffer& input);
  void SetOutput(Buffer* output);
  Buffer* GetOutput() { return output_; }
  const Buffer* GetOutput() const { return output_; }

  enum class OperationType : size_t {
    kImage = 0,
  };
  virtual OperationType GetOperationType() const = 0;

 protected:
  IOperation(const IOperation&) = default;
  IOperation& operator=(const IOperation&) = default;
  virtual absl::Status ProcessImpl(const Buffer& input) = 0;
  // Validate the input buffer. Return OK if the input buffer is valid for the
  // operation, otherwise return an error status.
  virtual absl::Status ValidateInput(const Buffer& input) const;
  // Validate the input and output buffers. Return OK if the input and output
  // buffers are valid for the operation, otherwise return an error status.
  absl::Status ValidateOrCreateOutput(const Buffer& input);
  // explicitly assign output buffer, otherwise it will be created automatically
  // Validate the output buffer. Return OK if the output buffer is valid for the
  // operation, otherwise return an error status.
  // This function is called only when the output buffer is not null.
  virtual absl::Status ValidateOutput(const Buffer& input) const = 0;
  // Create the output buffer. Return OK if the output buffer is created
  // successfully, otherwise return an error status.
  virtual absl::Status CreateOutput(const Buffer& input) = 0;

  bool output_assigned_ = false;
  Buffer* output_ = nullptr;
};
}  // namespace band

#endif