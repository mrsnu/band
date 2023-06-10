#ifndef BAND_BUFFER_BUFFER_PROCESSOR_H
#define BAND_BUFFER_BUFFER_PROCESSOR_H

#include "absl/status/statusor.h"
#include "band/buffer/operation.h"

namespace band {

class Buffer;
class IOperation;
class BufferProcessor;

class IBufferProcessorBuilder {
 public:
  IBufferProcessorBuilder() = default;
  virtual ~IBufferProcessorBuilder() = default;
  // Build a processor from the operations added to this builder.
  // The input and output buffers are used to validate the operations.
  // If the input and output buffers are nullptr, this builder only
  // validates the connections between operations.
  virtual absl::StatusOr<std::unique_ptr<BufferProcessor>> Build(
      const Buffer* input = nullptr, Buffer* output = nullptr) = 0;

  // Add an operation to the processor.
  // e.g., builder.AddOperation<OperationType>(args...);
  virtual absl::Status AddOperation(std::unique_ptr<IOperation> operation) {
    operations_.emplace_back(std::move(operation));
    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<BufferProcessor> CreateProcessor(
      std::vector<IOperation*> operations);

  IBufferProcessorBuilder(const IBufferProcessorBuilder&) = delete;
  IBufferProcessorBuilder& operator=(const IBufferProcessorBuilder&) = delete;

  std::vector<std::unique_ptr<IOperation>> operations_;
};

// A processor has a collection of sequential operations.
// The processor is responsible for validating the operations and
// executing them in the correct order.
// TODO(dostos)
// 1. Add a designated worker in an engine
// 2. User register a processor to the model
// 3. Engine should include pre/post processing task per each job, and
//    let a planner to include the pre/post processing time if needed.

class BufferProcessor {
 public:
  virtual ~BufferProcessor();
  absl::Status Process(const Buffer& input, Buffer& output);

 protected:
  friend class IBufferProcessorBuilder;
  BufferProcessor(std::vector<IOperation*> operations);
  BufferProcessor(const BufferProcessor&) = delete;
  BufferProcessor& operator=(const BufferProcessor&) = delete;

  std::vector<IOperation*> operations_;
};
}  // namespace band

#endif // BAND_BUFFER_BUFFER_PROCESSOR_H