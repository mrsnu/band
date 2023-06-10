#include "band/model.h"

#include "band/backend_factory.h"
#include "band/interface/model.h"
#include "band/logger.h"

#include "absl/strings/str_format.h"

namespace band {

ModelId Model::next_model_id_ = 0;
Model::Model() : model_id_(next_model_id_++) {}
Model::~Model() {}

ModelId Model::GetId() const { return model_id_; }

absl::Status Model::FromPath(BackendType backend_type, const char* filename) {
  if (GetBackendModel(backend_type)) {
    return absl::InternalError(
        absl::StrFormat("Tried to create %s model again for model id %d",
                        ToString(backend_type).c_str(), GetId()));
  }
  // TODO: check whether new model shares input / output shapes with existing
  // backend's model
  interface::IModel* backend_model =
      BackendFactory::CreateModel(backend_type, model_id_);

  if (!backend_model->FromPath(filename).ok()) {
    return absl::InternalError(
        absl::StrFormat("Failed to create %s model from %s",
                        ToString(backend_type).c_str(), filename));
  }
  backend_models_[backend_type] =
      std::shared_ptr<interface::IModel>(backend_model);
  return absl::OkStatus();
}

absl::Status Model::FromBuffer(BackendType backend_type, const char* buffer,
                               size_t buffer_size) {
  if (GetBackendModel(backend_type)) {
    return absl::InternalError(
        absl::StrFormat("Tried to create %s model again for model id %d",
                        ToString(backend_type), GetId()));
  }
  // TODO: check whether new model shares input / output shapes with existing
  // backend's model
  interface::IModel* backend_model =
      BackendFactory::CreateModel(backend_type, model_id_);
  if (backend_model == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "The given backend type `%s` is not registered in the binary.",
        ToString(backend_type)));
  }
  if (backend_model->FromBuffer(buffer, buffer_size).ok()) {
    backend_models_[backend_type] =
        std::shared_ptr<interface::IModel>(backend_model);
    return absl::OkStatus();
  } else {
    return absl::InternalError(absl::StrFormat(
        "Failed to create %s model from buffer", ToString(backend_type)));
  }
  return absl::OkStatus();
}

interface::IModel* Model::GetBackendModel(BackendType backend_type) {
  if (backend_models_.find(backend_type) != backend_models_.end()) {
    return backend_models_[backend_type].get();
  } else {
    return nullptr;
  }
}

std::set<BackendType> Model::GetSupportedBackends() const {
  std::set<BackendType> backends;
  for (auto it : backend_models_) {
    backends.insert(it.first);
  }
  return backends;
}
}  // namespace band