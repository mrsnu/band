#ifndef BAND_MODEL_H_
#define BAND_MODEL_H_

#include <map>

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

  BandStatus FromPath(BandBackendType backend_type, const char* filename);
  BandStatus FromBuffer(BandBackendType backend_type, const char* buffer,
                        size_t buffer_size);

  Interface::IModel* GetBackendModel(BandBackendType backend_type);
  std::set<BandBackendType> GetSupportedBackends() const;

 private:
  static ModelId next_model_id_;
  const ModelId model_id_;

  std::map<BandBackendType, Interface::IModel*> backend_models_;
};
}  // namespace Band

#endif