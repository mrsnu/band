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


#ifndef BAND_BACKEND_TFL_UTIL_H_
#define BAND_BACKEND_TFL_UTIL_H_

#include "band/common.h"
#include "band/interface/backend.h"

#include "tensorflow/lite/c/common.h"

#include "absl/status/status.h"

namespace band {
namespace tfl {

absl::Status GetBandStatus(TfLiteStatus status);
DataType GetBandDataType(TfLiteType type);

class TfLiteUtil : public interface::IBackendUtil {
 public:
  std::set<DeviceFlag> GetAvailableDevices() const override;
};

}  // namespace tfl
}  // namespace band

#endif