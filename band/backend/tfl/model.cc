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

#include "band/backend/tfl/model.h"

#include <iostream>

namespace band {
namespace tfl {
TfLiteModel::TfLiteModel(ModelId id) : interface::IModel(id) {}

BackendType TfLiteModel::GetBackendType() const { return BackendType::kTfLite; }

absl::Status TfLiteModel::FromPath(const char* filename) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ = tflite::FlatBufferModel::BuildFromFile(filename);
  path_ = filename;
  return flat_buffer_model_ ? absl::OkStatus()
                            : absl::InternalError("Cannot load from file.");
}

absl::Status TfLiteModel::FromBuffer(const char* buffer, size_t buffer_size) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ =
      tflite::FlatBufferModel::BuildFromBuffer(buffer, buffer_size);
  return flat_buffer_model_ ? absl::OkStatus()
                            : absl::InternalError("Cannot load from buffer.");
}

bool TfLiteModel::IsInitialized() const {
  return flat_buffer_model_ != nullptr;
}

}  // namespace tfl
}  // namespace band
