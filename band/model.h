#ifndef BAND_MODEL_H_
#define BAND_MODEL_H_

#include <map>
#include <memory>

#include "band/common.h"

namespace Band {
namespace Interface {
class IModel;
}

class Model {
 public:
  Model();
  ~Model();
  ModelId GetId() const;

  absl::Status FromPath(BackendType backend_type, const char* filename);
  absl::Status FromBuffer(BackendType backend_type, const char* buffer,
                        size_t buffer_size);

  Interface::IModel* GetBackendModel(BackendType backend_type);
  std::set<BackendType> GetSupportedBackends() const;

 private:
  static ModelId next_model_id_;
  const ModelId model_id_;

  std::map<BackendType, std::shared_ptr<Interface::IModel>> backend_models_;
};
}  // namespace Band

#endif