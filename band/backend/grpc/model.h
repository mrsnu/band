#ifndef BAND_BACKEND_GRPC_MODEL_H_
#define BAND_BACKEND_GRPC_MODEL_H_

#include "band/interface/model.h"

namespace band {
namespace grpc {

class GrpcModel : public interface::IModel {
 public:
  GrpcModel(ModelId id);
  BackendType GetBackendType() const override;
  absl::Status FromPath(const char* filename) override;
  absl::Status FromBuffer(const char* buffer, size_t buffer_size) override;
  bool IsInitialized() const override;
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_MODEL_H_