/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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