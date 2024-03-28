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

#ifndef BAND_BACKEND_TFL_MODEL_H_
#define BAND_BACKEND_TFL_MODEL_H_

#include <memory>

#include "band/interface/model.h"
#include "tensorflow/lite/model_builder.h"

namespace band {
namespace tfl {
class TfLiteModel : public interface::IModel {
 public:
  TfLiteModel(ModelId id);
  BackendType GetBackendType() const override;
  absl::Status FromPath(const char* filename) override;
  absl::Status FromBuffer(const char* buffer, size_t buffer_size) override;
  bool IsInitialized() const override;

  const tflite::FlatBufferModel* GetFlatBufferModel() const {
    return flat_buffer_model_.get();
  }

 private:
  std::unique_ptr<tflite::FlatBufferModel> flat_buffer_model_;
};
}  // namespace tfl
}  // namespace band

#endif