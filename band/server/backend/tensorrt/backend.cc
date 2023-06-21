#include "band/server/backend/tensorrt/backend.h"

namespace band {
namespace server {
namespace trt {

std::vector<const char*> kTensorRTLibPath = {
  "/usr/lib/x86_64-linux-gnu/libnvinfer.so",
};

TensorRTBackend::TensorRTBackend() {
  std::string trt_path = std::getenv("TENSORRT_LIB_PATH");
  if (trt_path.empty()) {
    BAND_LOG_PROD(BAND_LOG_WARNING, "TENSORRT_LIB_PATH is not set, using default path");
    for (const auto& path : kTensorRTLibPath) {
      loader_ = std::make_unique<TensorRTLoader>(path);
      if (loader_->IsInitialized()) {
        break;
      }
    }
  } else {
    loader_ = std::make_unique<TensorRTLoader>(trt_path.c_str());
  }
  
  if (!loader_->IsInitialized()) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to initialize TensorRT backend");
    return;
  }

  BAND_LOG_PROD(BAND_LOG_ERROR, "Successfully initialized TensorRT backend");
}

}  // namespace trt
}  // namespace server
}  // namespace band