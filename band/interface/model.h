#ifndef BAND_INTERFACE_MODEL_H_
#define BAND_INTERFACE_MODEL_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"

namespace Band {
namespace Interface {
/*
  Model interface for specific backend
*/
struct IModel : public IBackendSpecific {
 public:
  IModel(ModelId id) : id_(id) {}

  virtual BandStatus FromPath(const char* filename) = 0;
  virtual BandStatus FromBuffer(const char* buffer, size_t buffer_size) = 0;
  virtual bool IsInitialized() const = 0;
  ModelId GetId() const { return id_; }

 private:
  std::string path_;
  const ModelId id_;
};
}  // namespace Interface
}  // namespace Band

#endif