#include "band/backend/tfl/model.h"

#include <iostream>

namespace band {
namespace tfl {
TfLiteModel::TfLiteModel(ModelId id) : interface::IModel(id) {}

BackendType TfLiteModel::GetBackendType() const { return BackendType::kBandTfLite; }

absl::Status TfLiteModel::FromPath(const char* filename) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ = tflite::FlatBufferModel::BuildFromFile(filename);
  path_ = filename;
  return flat_buffer_model_ ? absl::OkStatus() : absl::InternalError("Cannot load from file.");
}

absl::Status TfLiteModel::FromBuffer(const char* buffer, size_t buffer_size) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ =
      tflite::FlatBufferModel::BuildFromBuffer(buffer, buffer_size);
  return flat_buffer_model_ ? absl::OkStatus() : absl::InternalError("Cannot load from buffer.");
}

bool TfLiteModel::IsInitialized() const {
  return flat_buffer_model_ != nullptr;
}

}  // namespace tfl
}  // namespace band
