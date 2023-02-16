#include "band/backend/tfl/model.h"

#include <iostream>

namespace Band {
namespace TfLite {
TfLiteModel::TfLiteModel(ModelId id) : Interface::IModel(id) {}

BackendType TfLiteModel::GetBackendType() const { return BackendType::TfLite; }

absl::Status TfLiteModel::FromPath(const char* filename) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ = tflite::FlatBufferModel::BuildFromFile(filename);
  path_ = filename;
  return flat_buffer_model_ ? absl::OkStatus() : kBandError;
}

absl::Status TfLiteModel::FromBuffer(const char* buffer, size_t buffer_size) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ =
      tflite::FlatBufferModel::BuildFromBuffer(buffer, buffer_size);
  return flat_buffer_model_ ? absl::OkStatus() : kBandError;
}

bool TfLiteModel::IsInitialized() const {
  return flat_buffer_model_ != nullptr;
}

}  // namespace TfLite
}  // namespace Band
