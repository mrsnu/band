#ifndef BAND_SERVER_BACKEND_TENSORRT_BACKEND_H_
#define BAND_SERVER_BACKEND_TENSORRT_BACKEND_H_

#include "band/server/backend/tensorrt/loader.h"

namespace band {
namespace server {
namespace trt {

class TensorRTBackend {
 public:
  TensorRTBackend();

 private:
  std::unique_ptr<TensorRTLoader> loader_;
};

}  // namespace trt
}  // namespace server
}  // namespace band

#endif  // BAND_SERVER_BACKEND_TENSORRT_BACKEND_H_