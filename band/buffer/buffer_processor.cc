// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/buffer/buffer_processor.h"

#include "band/logger.h"
#include "buffer_processor.h"

namespace band {

absl::Status BufferProcessor::Process(const Buffer& input, Buffer& output) {
  if (operations_.empty()) {
    return absl::InternalError("IProcessor: no operations are specified.");
  }

  // set the output buffer for the last operation
  operations_.back()->SetOutput(&output);

  Buffer const* next_input = &input;
  for (size_t i = 0; i < operations_.size(); ++i) {
    absl::Status status = operations_[i]->Process(*next_input);
    // skip cancelled status, and continue current input to the next operation
    if (!status.ok()) {
      BAND_LOG(LogSeverity::kError, "BufferProcessor::Process failed %s",
                    status.ToString().c_str());
      return status;
    }
    next_input = operations_[i]->GetOutput();
  }

  return absl::OkStatus();
}

BufferProcessor::BufferProcessor(std::vector<IBufferOperator*> operations)
    : operations_(std::move(operations))

{}

BufferProcessor::~BufferProcessor() {
  for (auto& operation : operations_) {
    delete operation;
  }
}

IBufferProcessorBuilder& IBufferProcessorBuilder::AddOperation(
    std::unique_ptr<IBufferOperator> operation) {
  operations_.emplace_back(std::move(operation));
  return *this;
}

std::unique_ptr<BufferProcessor> IBufferProcessorBuilder::CreateProcessor(
    std::vector<IBufferOperator*> operations) {
  return std::unique_ptr<BufferProcessor>(new BufferProcessor(operations));
}

}  // namespace band