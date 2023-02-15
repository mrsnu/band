#ifndef BAND_MODEL_H_
#define BAND_MODEL_H_

#include <map>
#include <memory>

#include "band/common.h"

namespace band {
namespace interface {
class IModel;
}

class Model {
 public:
  Model();
  ~Model();
  ModelId GetId() const;

  BandStatus FromPath(BandBackendType backend_type, const char* filename);
  BandStatus FromBuffer(BandBackendType backend_type, const char* buffer,
                        size_t buffer_size);

  interface::IModel* GetBackendModel(BandBackendType backend_type);
  std::set<BandBackendType> GetSupportedBackends() const;

 private:
  static ModelId next_model_id_;
  const ModelId model_id_;

  std::map<BandBackendType, std::shared_ptr<interface::IModel>> backend_models_;
};
}  // namespace band

#endif