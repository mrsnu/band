#ifndef BAND_BUFFER_COMMON_OPERATION_H_
#define BAND_BUFFER_COMMON_OPERATION_H_

#include "absl/status/status.h"
#include "band/buffer/buffer.h"
#include "band/buffer/operator.h"

namespace band {
namespace buffer {

class Normalize : public IBufferOperator {
 public:
  Normalize(float mean, float std, bool inplace)
      : mean_(mean), std_(std), inplace_(inplace) {}

  virtual IBufferOperator* Clone() const override;
  virtual Type GetOpType() const override;

  void SetOutput(Buffer* output);

 protected:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

  float mean_, std_;
  bool inplace_;
};

// DataTypeConvert is equivalent to Normalize(0.f, 1.f) without inplace.
// It automatically converts the internal data type to output data type.
class DataTypeConvert : public Normalize {
 public:
  DataTypeConvert() : Normalize(0.f, 1.f, false) {}

  virtual IBufferOperator* Clone() const override;

 private:
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
};

}  // namespace buffer
}  // namespace band

#endif  // BAND_BUFFER_COMMON_OPERATION_H_