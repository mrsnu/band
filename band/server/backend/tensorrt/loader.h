#ifndef BAND_SERVER_BACKEND_TENSORRT_LOADER_H_
#define BAND_SERVER_BACKEND_TENSORRT_LOADER_H_

#include "band/logger.h"
#include "band/server/util/dso_loader.h"

#include "third_party/tensorrt/NvInfer.h"

typedef void* (*PFN_createInferBuilder_INTERNAL)(void*, int);
typedef void* (*PFN_createInferRefitter_INTERNAL)(void*, void*, int);
typedef void* (*PFN_createInferRuntime_INTERNAL)(void*, int);

#define DEFINE_SYMBOL(symbol) PFN_##symbol symbol##_ = nullptr;

#define LOAD_SYMBOL(symbol)                               \
  symbol##_ = LoadSymbol<PFN_##symbol>(handle_, #symbol); \
  if (symbol##_ == nullptr) {                             \
    return;                                               \
  }

#define DEFINE_TRT_SYMBOLS()                   \
  DEFINE_SYMBOL(createInferBuilder_INTERNAL);  \
  DEFINE_SYMBOL(createInferRefitter_INTERNAL); \
  DEFINE_SYMBOL(createInferRuntime_INTERNAL);

#define LOAD_TRT_SYMBOLS()                   \
  LOAD_SYMBOL(createInferBuilder_INTERNAL);  \
  LOAD_SYMBOL(createInferRefitter_INTERNAL); \
  LOAD_SYMBOL(createInferRuntime_INTERNAL);

namespace band {
namespace server {
namespace trt {

class TensorRTLoader : public DsoLoader {
 public:
  TensorRTLoader(const char* path) {
    auto status_or_handle = Load(path);
    if (!status_or_handle.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "%s", status_or_handle.status().message());
      return;
    }
    handle_ = status_or_handle.value();
    LOAD_TRT_SYMBOLS();
    initialized_ = true;
  }

  ~TensorRTLoader() {
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
  DEFINE_TRT_SYMBOLS()
};

}  // namespace trt
}  // namespace server
}  // namespace band

#endif  // BAND_SERVER_BACKEND_TENSORRT_LOADER_H_