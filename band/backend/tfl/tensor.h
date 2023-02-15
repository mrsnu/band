#ifndef BAND_BACKEND_TFL_TENSOR_H_
#define BAND_BACKEND_TFL_TENSOR_H_

#include "band/interface/tensor_view.h"
#include "tensorflow/lite/c/common.h"

namespace band {
namespace tfl {
class TfLiteTensorView : public interface::ITensorView {
 public:
  TfLiteTensorView(TfLiteTensor* tensor);

  BandBackendType GetBackendType() const override;
  BandType GetType() const override;
  void SetType(BandType type) override;
  const char* GetData() const override;
  char* GetData() override;
  const int* GetDims() const override;
  size_t GetNumDims() const override;
  void SetDims(const std::vector<int>& dims) override;
  size_t GetBytes() const override;
  const char* GetName() const override;
  BandQuantization GetQuantization() const override;
  void SetQuantization(BandQuantization quantization) override;

 private:
  TfLiteTensor* tensor_ = nullptr;
};
}  // namespace tfl
}  // namespace band

#endif