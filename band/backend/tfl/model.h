#ifndef BAND_BACKEND_TFL_MODEL_H_
#define BAND_BACKEND_TFL_MODEL_H_

#include <memory>

#include "band/interface/model.h"
#include "tensorflow/lite/model_builder.h"

namespace Band {
namespace TfLite {
class TfLiteModel : public Interface::IModel {
 public:
  TfLiteModel(ModelId id);
  BackendType GetBackendType() const override;
  BandStatus FromPath(const char* filename) override;
  BandStatus FromBuffer(const char* buffer, size_t buffer_size) override;
  bool IsInitialized() const override;

  const tflite::FlatBufferModel* GetFlatBufferModel() const {
    return flat_buffer_model_.get();
  }

 private:
  std::unique_ptr<tflite::FlatBufferModel> flat_buffer_model_;
};
}  // namespace TfLite
}  // namespace Band

#endif