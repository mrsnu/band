#include "band/tensor/processor.h"

#include "processor.h"

namespace band {
namespace tensor {

absl::Status IProcessor::Process(const Buffer& input, Buffer& output) {
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

IProcessor::IProcessor(std::vector<IOperation*> operations)
    : operations_(std::move(operations))

{}

IProcessor::~IProcessor() {
  for (auto& operation : operations_) {
    delete operation;
  }
}

std::unique_ptr<IProcessor> IProcessorBuilder::CreateProcessor(
    std::vector<IOperation*> operations) {
  return std::unique_ptr<IProcessor>(new IProcessor(operations));
}

}  // namespace tensor
}  // namespace band