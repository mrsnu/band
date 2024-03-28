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

#ifndef BAND_INTERFACE_MODEL_H_
#define BAND_INTERFACE_MODEL_H_

#include <string>
#include <vector>

#include "band/common.h"
#include "band/interface/backend.h"

#include "absl/status/status.h"

namespace band {
namespace interface {
/*
  Model interface for specific backend
*/
struct IModel : public IBackendSpecific {
 public:
  IModel(ModelId id) : id_(id) {}
  virtual ~IModel() = default;

  virtual absl::Status FromPath(const char* filename) = 0;
  virtual absl::Status FromBuffer(const char* buffer, size_t buffer_size) = 0;
  virtual bool IsInitialized() const = 0;
  ModelId GetId() const { return id_; }
  const std::string& GetPath() const { return path_; }

 protected:
  std::string path_;
  const ModelId id_;
};
}  // namespace interface
}  // namespace band

#endif