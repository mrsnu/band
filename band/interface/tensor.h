#ifndef BAND_INTERFACE_TENSOR_H_
#define BAND_INTERFACE_TENSOR_H_

#include <vector>

#include "band/common.h"

namespace band {
namespace interface {
struct ITensor {
 public:
  virtual ~ITensor() = default;

  virtual DataType GetType() const = 0;
  virtual void SetType(DataType type) = 0;
  virtual const char* GetData() const = 0;
  virtual char* GetData() = 0;
  virtual const int* GetDims() const = 0;
  virtual size_t GetNumDims() const = 0;
  size_t GetNumElements() const;
  std::vector<int> GetDimsVector() const;
  virtual void SetDims(const std::vector<int>& dims) = 0;
  virtual size_t GetBytes() const = 0;
  virtual const char* GetName() const = 0;
  virtual Quantization GetQuantization() const = 0;
  virtual void SetQuantization(Quantization quantization) = 0;
  bool operator==(const ITensor& rhs) const;
  bool operator!=(const ITensor& rhs) const;

  absl::Status CopyDataFrom(const ITensor& rhs);
  absl::Status CopyDataFrom(const ITensor* rhs);
};
}  // namespace interface
}  // namespace band

#endif