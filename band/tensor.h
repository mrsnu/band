#ifndef BAND_TENSOR_H_
#define BAND_TENSOR_H_

#include "band/c/common.h"
#include "band/interface/tensor.h"

#include <string>
#include <vector>

namespace Band {
/*
  Tensor interface that tensor view / band tensor shares
*/

class Tensor : public Interface::ITensor {
public:
  explicit Tensor(Interface::ITensor *tensor_view);
  ~Tensor();

  BandType GetType() const override;
  void SetType(BandType type) override;
  const char *GetData() const override;
  char *GetData() override;
  std::vector<int> GetDims() const override;
  void SetDims(const std::vector<int> &dims) override;
  size_t GetBytes() const override;
  const char *GetName() const override;
  BandQuantization GetQuantization() const override;
  void SetQuantization(BandQuantization quantization) override;

private:
  BandType type_;
  BandQuantization quantization_;
  size_t num_bytes_;
  std::vector<int> dims_;
  char *data_;
  std::string name_;
};
} // namespace Band

#endif // BAND_TENSOR_H_
