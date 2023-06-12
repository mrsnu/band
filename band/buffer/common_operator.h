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

 private:
  virtual absl::Status ProcessImpl(const Buffer& input) override;
  virtual absl::Status ValidateInput(const Buffer& input) const override;
  virtual absl::Status ValidateOutput(const Buffer& input) const override;
  virtual absl::Status CreateOutput(const Buffer& input) override;

  float mean_, std_;
  bool inplace_;
};

}  // namespace buffer
}  // namespace band

#endif  // BAND_BUFFER_COMMON_OPERATION_H_