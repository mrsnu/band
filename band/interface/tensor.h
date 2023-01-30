#ifndef BAND_INTERFACE_TENSOR_H_
#define BAND_INTERFACE_TENSOR_H_

#include <vector>

#include "band/c/common.h"

namespace Band {
namespace Interface {
struct ITensor {
 public:
  virtual ~ITensor() = default;

  virtual BandType GetType() const = 0;
  virtual void SetType(BandType type) = 0;
  virtual const char* GetData() const = 0;
  virtual char* GetData() = 0;
  virtual const int* GetDims() const = 0;
  virtual size_t GetNumDims() const = 0;
  size_t GetNumElements() const;
  std::vector<int> GetDimsVector() const;
  virtual void SetDims(const std::vector<int>& dims) = 0;
  virtual size_t GetBytes() const = 0;
  virtual const char* GetName() const = 0;
  virtual BandQuantization GetQuantization() const = 0;
  virtual void SetQuantization(BandQuantization quantization) = 0;
  bool operator==(const ITensor& rhs) const;

  BandStatus CopyDataFrom(const ITensor& rhs);
  BandStatus CopyDataFrom(const ITensor* rhs);
};
}  // namespace Interface
}  // namespace Band

#endif