#ifndef BAND_TENSOR_TENSOR_H_
#define BAND_TENSOR_TENSOR_H_

#include <string>
#include <vector>

#include "band/interface/tensor.h"

namespace band {
/*
  Tensor interface that tensor view / band tensor shares
*/
class ExternalBuffer;
class Tensor : public interface::ITensor {
 public:
  // deep copy from tensor view
  explicit Tensor(interface::ITensor* tensor_view);
  explicit Tensor(ExternalBuffer* external_buffer);
  ~Tensor();

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

 private:
  DataType type_;
  Quantization quantization_;
  size_t num_bytes_;
  std::vector<int> dims_;
  char* data_;
  std::string name_;
};
}  // namespace band

#endif  // BAND_TENSOR_TENSOR_H_
