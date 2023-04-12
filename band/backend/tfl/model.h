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

#endif  // BAND_BACKEND_TFL_MODEL_H_