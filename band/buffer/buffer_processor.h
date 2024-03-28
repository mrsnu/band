/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_BUFFER_BUFFER_PROCESSOR_H_
#define BAND_BUFFER_BUFFER_PROCESSOR_H_

#include "absl/status/statusor.h"
#include "band/buffer/operator.h"

namespace band {

class Buffer;
class IBufferOperator;
class BufferProcessor;

class IBufferProcessorBuilder {
 public:
  IBufferProcessorBuilder() = default;
  virtual ~IBufferProcessorBuilder() = default;
  // Build a processor from the operations added to this builder.
  virtual absl::StatusOr<std::unique_ptr<BufferProcessor>> Build() = 0;

  // Add an operation to the processor.
  // e.g., builder.AddOperation<Type>(args...);
  IBufferProcessorBuilder& AddOperation(
      std::unique_ptr<IBufferOperator> operation);

 protected:
  std::unique_ptr<BufferProcessor> CreateProcessor(
      std::vector<IBufferOperator*> operations);

  IBufferProcessorBuilder(const IBufferProcessorBuilder&) = delete;
  IBufferProcessorBuilder& operator=(const IBufferProcessorBuilder&) = delete;

  std::vector<std::unique_ptr<IBufferOperator>> operations_;
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
  virtual absl::Status Process(const Buffer& input, Buffer& output);

 protected:
  friend class IBufferProcessorBuilder;
  BufferProcessor(std::vector<IBufferOperator*> operations);
  BufferProcessor(const BufferProcessor&) = delete;
  BufferProcessor& operator=(const BufferProcessor&) = delete;

  std::vector<IBufferOperator*> operations_;
};
}  // namespace band

#endif  // BAND_BUFFER_BUFFER_PROCESSOR_H_