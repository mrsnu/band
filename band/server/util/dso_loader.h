#ifndef BAND_SERVER_BACKEND_TENSORRT_DSO_LOADER_H_
#define BAND_SERVER_BACKEND_TENSORRT_DSO_LOADER_H_

#include "absl/status/statusor.h"

namespace band {
namespace server {
  
class DsoLoader {
 protected:
  absl::StatusOr<void*> Load(const char* path);
  absl::StatusOr<void*> GetSymbol(void* handle, const char* symbol);
  void Unload(void* handle);
};

}  // namespace server
}  // namespace band

#endif  // BAND_SERVER_BACKEND_TENSORRT_DSO_LOADER_H_
