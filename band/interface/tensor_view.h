#ifndef BAND_INTERFACE_TENSOR_VIEW_H_
#define BAND_INTERFACE_TENSOR_VIEW_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"
#include "band/interface/tensor.h"

namespace Band {
namespace Interface {
/*
  Tensor view interface for communication with
  backend specific / owned tensor types.

  Band::Tensor
    <- (deep cpy) Band::ITensorView view
    <- (shallow cpy) Backend tensor
*/
struct ITensorView : public IBackendSpecific, public ITensor {};

} // namespace Interface
} // namespace Band

#endif