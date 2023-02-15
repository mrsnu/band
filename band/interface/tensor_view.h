#ifndef BAND_INTERFACE_TENSOR_VIEW_H_
#define BAND_INTERFACE_TENSOR_VIEW_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"
#include "band/interface/tensor.h"

namespace band {
namespace interface {
/*
  Tensor view interface for communication with
  backend specific / owned tensor types.

  band::Tensor
    <- (deep cpy) band::ITensorView view
    <- (shallow cpy) Backend tensor
*/
struct ITensorView : public IBackendSpecific, public ITensor {};

}  // namespace interface
}  // namespace band

#endif