#include "band/buffer/buffer_processor.h"

#include "buffer_processor.h"

namespace band {

absl::Status BufferProcessor::Process(const Buffer& input, Buffer& output) {
  if (operations_.empty()) {
    return absl::InternalError("IProcessor: no operations are specified.");
  }

  // TODO(dostos): currently, only the last operation is allowed to implicitly
  // infer parameters from the propagated input buffer and a given output
  // buffer. We should add additional back-to-front propagation of parameters
  // e.g., buffer size, color space, etc. to allow for more automatic config
  // such as the following:
  //   input -> <resize (output)> -> <color conversion (output)> -> output

  // set the output buffer for the last operation
  operations_.back()->SetOutput(&output);

  Buffer const* next_input = &input;
  for (size_t i = 0; i < operations_.size(); ++i) {
    RETURN_IF_ERROR(operations_[i]->Process(*next_input));
    next_input = operations_[i]->GetOutput();
  }

  return absl::OkStatus();
}

BufferProcessor::BufferProcessor(std::vector<IOperation*> operations)
    : operations_(std::move(operations))

{}

BufferProcessor::~BufferProcessor() {
  for (auto& operation : operations_) {
    delete operation;
  }
}

std::unique_ptr<BufferProcessor> IBufferProcessorBuilder::CreateProcessor(
    std::vector<IOperation*> operations) {
  return std::unique_ptr<BufferProcessor>(new BufferProcessor(operations));
}

}  // namespace band