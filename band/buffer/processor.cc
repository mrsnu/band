#include "band/buffer/processor.h"

#include "processor.h"

namespace band {

absl::Status Processor::Process(const Buffer& input, Buffer& output) {
  if (operations_.empty()) {
    return absl::InternalError("IProcessor: no operations are specified.");
  }

  operations_.back()->SetOutput(&output);

  for (size_t i = 0; i < operations_.size(); ++i) {
    absl::Status status = operations_[i]->Process(input);
    if (!status.ok()) {
      return status;
    }
    if (i + 1 < operations_.size()) {
      operations_[i + 1]->SetOutput(operations_[i]->GetOutput());
    }
  }

  return absl::OkStatus();
}

Processor::Processor(std::vector<IOperation*> operations)
    : operations_(std::move(operations))

{}

Processor::~Processor() {
  for (auto& operation : operations_) {
    delete operation;
  }
}

std::unique_ptr<Processor> IProcessorBuilder::CreateProcessor(
    std::vector<IOperation*> operations) {
  return std::unique_ptr<Processor>(new Processor(operations));
}

}  // namespace band