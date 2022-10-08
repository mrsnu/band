#include "band/backend/tfl/model.h"

#include <iostream>

#include "band/c/common.h"

namespace Band {
namespace TfLite {
TfLiteModel::TfLiteModel(ModelId id) : Interface::IModel(id) {}

BandBackendType TfLiteModel::GetBackendType() const { return kBandTfLite; }

absl::Status TfLiteModel::FromPath(const char* filename) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ = tflite::FlatBufferModel::BuildFromFile(filename);
  return flat_buffer_model_
             ? absl::OkStatus()
             : absl::InternalError("Converting model from a file failed.");
}

absl::Status TfLiteModel::FromBuffer(const char* buffer, size_t buffer_size) {
  // TODO: Add Band TFLBackend error reporter
  flat_buffer_model_ =
      tflite::FlatBufferModel::BuildFromBuffer(buffer, buffer_size);
  return flat_buffer_model_
             ? absl::OkStatus()
             : absl::InternalError("Converting model from a buffer failed.");
}

bool TfLiteModel::IsInitialized() const {
  return flat_buffer_model_ != nullptr;
}

}  // namespace TfLite
}  // namespace Band
