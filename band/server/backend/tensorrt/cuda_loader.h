#ifndef BAND_SERVER_BACKEND_CUDA_LOADER_H_
#define BAND_SERVER_BACKEND_CUDA_LOADER_H_

#include "band/logger.h"
#include "band/server/util/dso_loader.h"

#define DEFINE_SYMBOL(symbol) PFN_##symbol symbol##_ = nullptr;

#define LOAD_SYMBOL(symbol)                               \
  symbol##_ = LoadSymbol<PFN_##symbol>(handle_, #symbol); \
  if (symbol##_ == nullptr) {                             \
    return;                                               \
  }

#define DEFINE_CUDA_SYMBOLS() \
  DEFINE_SYMBOL();

#define LOAD_CUDA_SYMBOLS() \
  LOAD_SYMBOL();

namespace band {
namespace server {
namespace cuda {

class CudaLoader : public DsoLoader {
 public:
  CudaLoader(const char* path) {
    auto status_or_handle = Load(path);
    if (!status_or_handle.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "%s", status_or_handle.status().message());
      return;
    }
    handle_ = status_or_handle.value();
    LOAD_CUDA_SYMBOLS();
    initialized_ = true;
  }

  ~CudaLoader() {
    if (handle_) {
      Unload(handle_);
    }
  }

  bool IsInitialized() { return initialized_; }

 private:
  template <typename T>
  T LoadSymbol(void* handle, const char* symbol) {
    auto status_or_symbol = GetSymbol(handle, symbol);
    if (!status_or_symbol.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "%s", status_or_symbol.status().message());
      return nullptr;
    }
    return reinterpret_cast<T>(status_or_symbol.value());
  }

  bool initialized_ = false;
  void* handle_ = nullptr;
  DEFINE_CUDA_SYMBOLS()
};

}  // namespace cuda
}  // namespace server
}  // namespace band

#endif  // BAND_SERVER_BACKEND_CUDA_LOADER_H_