#include "band/buffer/operation.h"

#include "band/logger.h"
#include "operation.h"

namespace band {
IOperation::~IOperation() {
  if (output_ != nullptr && !output_assigned_) {
    delete output_;
  }
}

absl::Status IOperation::Process(const Buffer& input) {
  RETURN_IF_ERROR(ValidateInput(input));
  RETURN_IF_ERROR(ValidateOrCreateOutput(input));
  RETURN_IF_ERROR(ProcessImpl(input));
  return absl::OkStatus();
}

void IOperation::SetOutput(Buffer* output) {
  if (!output_assigned_) {
    delete output_;
  }
  output_ = output;
  output_assigned_ = true;
}

absl::Status IOperation::ValidateInput(const Buffer& input) const {
  return absl::OkStatus();
}

absl::Status IOperation::ValidateOrCreateOutput(const Buffer& input) {
  absl::Status status = output_ == nullptr
                            ? absl::InternalError("Null output buffer")
                            : ValidateOutput(input);
  if (!status.ok()) {
    if (output_assigned_) {
      // if output is externally assigned, return the error
      return status;
    } else {
      // if output is not externally assigned, try to create it
      RETURN_IF_ERROR(CreateOutput(input));
      // validate the created output
      status = ValidateOutput(input);
      if (!status.ok()) {
        // real trouble if the created output is not valid
        BAND_LOG_PROD(BAND_LOG_ERROR,
                      "Failed to create valid output buffer: %s",
                      status.message());
      }
      return status;
    }
  }
  return absl::OkStatus();
}
}  // namespace band
