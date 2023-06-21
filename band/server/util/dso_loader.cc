#include "band/server/util/dso_loader.h"

#include <dlfcn.h>

namespace band {
namespace server {

absl::StatusOr<void*> DsoLoader::Load(const char* path) {
  void* handle = dlopen(path, RTLD_LAZY);
  if (handle == nullptr) {
    return absl::InternalError(dlerror());
  }
  return handle;
}

absl::StatusOr<void*> DsoLoader::GetSymbol(void* handle, const char* symbol) {
  if (handle == nullptr) {
    return absl::InternalError("handle is nullptr");
  }
  void* sym = dlsym(handle, symbol);
  if (sym == nullptr) {
    return absl::InternalError(dlerror());
  }
  return sym;
}

void DsoLoader::Unload(void* handle) { 
  if (handle != nullptr) {
    dlclose(handle);
  }
}

}  // namespace server
}  // namespace band
