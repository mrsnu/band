#include "band/model.h"

#include "absl/strings/str_format.h"
#include "band/backend_factory.h"
#include "band/interface/model.h"
#include "band/logger.h"


/**
 * @file model.cc
 * @brief Implementation of the Model class.
 */

namespace band {

ModelId Model::next_model_id_ = 0;

/**
 * @brief Default constructor for the Model class.
 * @details Initializes the model_id_ member variable using the next_model_id_ static variable.
 */
Model::Model() : model_id_(next_model_id_++) {}

/**
 * @brief Destructor for the Model class.
 */
Model::~Model() {}

/**
 * @brief Get the ID of the model.
 * @return The ID of the model.
 */
ModelId Model::GetId() const { return model_id_; }

/**
 * @brief Load a model from a file path.
 * @param backend_type The type of the backend.
 * @param filename The path to the model file.
 * @return The status of the operation.
 * @retval absl::OkStatus The model was loaded successfully.
 * @retval absl::InternalError An error occurred while loading the model.
 */
absl::Status Model::FromPath(BackendType backend_type, const char* filename) {
  if (GetBackendModel(backend_type)) {
    return absl::InternalError(
        absl::StrFormat("Tried to create %s model again for model id %d",
                        ToString(backend_type), GetId()));
  }
  // TODO: check whether new model shares input / output shapes with existing
  // backend's model
  interface::IModel* backend_model =
      BackendFactory::CreateModel(backend_type, model_id_);

  if (!backend_model->FromPath(filename).ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to create %s model from %s", ToString(backend_type), filename));
  }
  backend_models_[backend_type] =
      std::shared_ptr<interface::IModel>(backend_model);
  return absl::OkStatus();
}

/**
 * @brief Load a model from a buffer.
 * @param backend_type The type of the backend.
 * @param buffer The buffer containing the model data.
 * @param buffer_size The size of the buffer.
 * @return The status of the operation.
 * @retval absl::OkStatus The model was loaded successfully.
 * @retval absl::InternalError An error occurred while loading the model.
 */
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

/**
 * @brief Get the backend model associated with the given backend type.
 * @param backend_type The type of the backend.
 * @return A pointer to the backend model, or nullptr if not found.
 */
interface::IModel* Model::GetBackendModel(BackendType backend_type) {
  if (backend_models_.find(backend_type) != backend_models_.end()) {
    return backend_models_[backend_type].get();
  } else {
    return nullptr;
  }
}

/**
 * @brief Get the set of supported backend types.
 * @return The set of supported backend types.
 */
std::set<BackendType> Model::GetSupportedBackends() const {
  std::set<BackendType> backends;
  for (auto it : backend_models_) {
    backends.insert(it.first);
  }
  return backends;
}

}  // namespace band