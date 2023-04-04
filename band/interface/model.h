#ifndef BAND_INTERFACE_MODEL_H_
#define BAND_INTERFACE_MODEL_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"

#include "absl/status/status.h"

namespace band {
namespace interface {
/*
  Model interface for specific backend
*/
struct IModel : public IBackendSpecific {
 public:
  IModel(ModelId id) : id_(id) {}

  virtual absl::Status FromPath(const char* filename) = 0;
  virtual absl::Status FromBuffer(const char* buffer, size_t buffer_size) = 0;
  virtual bool IsInitialized() const = 0;
  ModelId GetId() const { return id_; }
  const std::string& GetPath() const { return path_; }

 protected:
  std::string path_;
  const ModelId id_;
};
}  // namespace interface
}  // namespace band

#endif