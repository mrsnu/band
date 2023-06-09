#ifndef BAND_TENSOR_PROCESSOR_H
#define BAND_TENSOR_PROCESSOR_H

#include "absl/status/statusor.h"
#include "band/tensor/operation.h"

namespace band {
namespace tensor {

class IProcessor;
class IOperation;
class Buffer;

class IProcessorBuilder {
 public:
  IProcessorBuilder() = default;
  virtual ~IProcessorBuilder() = default;
  // Build a processor from the operations added to this builder.
  // The input and output buffers are used to validate the operations.
  // If the input and output buffers are nullptr, this builder only
  // validates the connections between operations.
  virtual absl::StatusOr<std::unique_ptr<IProcessor>> Build(
      const Buffer* input = nullptr, Buffer* output = nullptr) = 0;

  // Add an operation to the processor.
  // e.g., builder.AddOperation<OperationType>(args...);
  virtual absl::Status AddOperation(std::unique_ptr<IOperation> operation) {
    operations_.emplace_back(std::move(operation));
    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<IProcessor> CreateProcessor(
      std::vector<IOperation*> operations);

  IProcessorBuilder(const IProcessorBuilder&) = delete;
  IProcessorBuilder& operator=(const IProcessorBuilder&) = delete;

  std::vector<std::unique_ptr<IOperation>> operations_;
};

// A processor is a collection of operations.
// The processor is responsible for validating the operations and
// executing them in the correct order.
// TODO(dostos)
// 1. Add a designated worker in an engine
// 2. User register a processor to the model
// 3. Engine should include pre/post processing task per each job, and
//    let a planner to include the pre/post processing time if needed.

class IProcessor {
 public:
  virtual ~IProcessor();
  absl::Status Process(const Buffer& input, Buffer& output);

 protected:
  friend class IProcessorBuilder;
  IProcessor(std::vector<IOperation*> operations);
  IProcessor(const IProcessor&) = delete;
  IProcessor& operator=(const IProcessor&) = delete;

  std::vector<IOperation*> operations_;
};

}  // namespace tensor
}  // namespace band

#endif