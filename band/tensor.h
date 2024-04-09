#ifndef BAND_TENSOR_H_
#define BAND_TENSOR_H_

#include <string>
#include <vector>

#include "band/interface/tensor.h"

namespace band {
/*
  Tensor interface that tensor view / band tensor shares
*/
class Tensor : public interface::ITensor {
          // 用于管理和操作在计算任务中常用的张量数据
 public:
  explicit Tensor(interface::ITensor* tensor_view, bool copy_data = false);
  ~Tensor();

  DataType GetType() const override;
  void SetType(DataType type) override;
  const char* GetData() const override;
  char* GetData() override;
  const int* GetDims() const override;
  size_t GetNumDims() const override;
  void SetDims(const std::vector<int>& dims) override;
  const char* GetName() const override;
  Quantization GetQuantization() const override;
  absl::Status SetQuantization(Quantization quantization) override;

 private:
  DataType type_;
  Quantization quantization_;
  std::vector<int> dims_;
  char* data_;
  std::string name_;
};
}  // namespace band

#endif  // BAND_TENSOR_H_
