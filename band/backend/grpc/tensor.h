#ifndef BAND_BACKEND_GRPC_TENSOR_H_
#define BAND_BACKEND_GRPC_TENSOR_H_

#include "band/interface/tensor_view.h"

namespace band {
namespace grpc {

class GrpcTensorView : public interface::ITensorView  {
 public:
  BackendType GetBackendType() const override;
  DataType GetType() const override;
  void SetType(DataType type) override;
  const char* GetData() const override;
  char* GetData() override;
  const int* GetDims() const override;
  size_t GetNumDims() const override;
  void SetDims(const std::vector<int>& dims) override;
  size_t GetBytes() const override;
  const char* GetName() const override;
  Quantization GetQuantization() const override;
  absl::Status SetQuantization(Quantization quantization) override;
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_TENSOR_H_