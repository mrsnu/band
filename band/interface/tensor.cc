// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/interface/tensor.h"

#include <cstring>

#include "band/logger.h"
#include "tensor.h"

namespace band {
namespace interface {

bool ITensor::operator==(const ITensor& rhs) const {
  if (GetType() != rhs.GetType()) {
    return false;
  }

  if (GetDimsVector() != rhs.GetDimsVector()) {
    return false;
  }

  return true;
}

bool ITensor::operator!=(const ITensor& rhs) const { return !(*this == rhs); }

size_t ITensor::GetBytes() const {
  return GetDataTypeBytes(GetType()) * GetNumElements();
}

size_t ITensor::GetNumElements() const {
  size_t num_elements = 1;
  for (auto dim : GetDimsVector()) {
    num_elements *= dim;
  }
  return num_elements;
}

std::vector<int> ITensor::GetDimsVector() const {
  return std::vector<int>(GetDims(), GetDims() + GetNumDims());
}

absl::Status ITensor::CopyDataFrom(const ITensor& rhs) {
  if (*this != rhs) {
    return absl::InternalError("");
  }
  memcpy(GetData(), rhs.GetData(), GetBytes());
  return absl::OkStatus();
}

absl::Status ITensor::CopyDataFrom(const ITensor* rhs) {
  if (!rhs) {
    return absl::InternalError("Tried to copy null tensor");
  }

  return CopyDataFrom(*rhs);
}
}  // namespace interface
}  // namespace band