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

#ifndef BAND_INTERFACE_BACKEND_H_
#define BAND_INTERFACE_BACKEND_H_

namespace band {
namespace interface {
class IBackendSpecific {
 public:
  virtual BackendType GetBackendType() const = 0;
  bool IsCompatible(const IBackendSpecific& rhs) const {
    return IsCompatible(&rhs);
  }
  bool IsCompatible(const IBackendSpecific* rhs) const {
    return GetBackendType() == rhs->GetBackendType();
  }
};

class IBackendUtil {
 public:
  virtual std::set<DeviceFlag> GetAvailableDevices() const = 0;
};

}  // namespace interface
}  // namespace band

#endif  // BAND_INTERFACE_BACKEND_H_