#include "band/model.h"
#include "band/interface/backend_factory.h"
#include "band/interface/model.h"
#include "band/logger.h"

namespace Band {

ModelId Model::next_model_id_ = 0;
Model::Model() : model_id_(next_model_id_++) {}
Model::~Model() {
  for (auto it : backend_models_) {
    delete it.second;
  }
}
ModelId Model::GetId() const { return model_id_; }

BandStatus Model::FromPath(BandBackendType backend_type, const char *filename) {
  if (GetBackendModel(backend_type)) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Tried to create %s model again for model id %d",
                      BandBackendGetName(backend_type), GetId());
    return kBandError;
  }
  // TODO: check whether new model shares input / output shapes with existing
  // backend's model
  Interface::IModel *backend_model =
      Interface::BackendFactory::CreateModel(backend_type, model_id_);
  if (backend_model->FromPath(filename) == kBandOk) {
    backend_models_[backend_type] = backend_model;
    return kBandOk;
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Failed to create %s model from %s",
                      BandBackendGetName(backend_type), filename);
    return kBandError;
  }
}

BandStatus Model::FromBuffer(BandBackendType backend_type, const char *buffer,
                             size_t buffer_size) {
  if (GetBackendModel(backend_type)) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Tried to create %s model again for model id %d",
                      BandBackendGetName(backend_type), GetId());
    return kBandError;
  }
  // TODO: check whether new model shares input / output shapes with existing
  // backend's model
  Interface::IModel *backend_model =
      Interface::BackendFactory::CreateModel(backend_type, model_id_);
  if (backend_model->FromBuffer(buffer, buffer_size) == kBandOk) {
    backend_models_[backend_type] = backend_model;
    return kBandOk;
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Failed to create %s model from buffer",
                      BandBackendGetName(backend_type));
    return kBandError;
  }
}

Interface::IModel *Model::GetBackendModel(BandBackendType backend_type) {
  if (backend_models_.find(backend_type) != backend_models_.end()) {
    return backend_models_[backend_type];
  } else {
    return nullptr;
  }
}

std::set<BandBackendType> Model::GetSupportedBackends() const {
  std::set<BandBackendType> backends;
  for (auto it : backend_models_) {
    backends.insert(it.first);
  }
  return backends;
}
} // namespace Band