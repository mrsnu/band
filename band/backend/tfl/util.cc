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

#include "band/backend/tfl/util.h"

#include <mutex>
#include <set>

#include "band/backend/tfl/model_executor.h"
#include "band/logger.h"

namespace band {
namespace tfl {

absl::Status GetBandStatus(TfLiteStatus status) {
  if (status == kTfLiteOk) {
    return absl::OkStatus();
  } else {
    return absl::InternalError("TfLite Error");
  }
}

DataType GetBandDataType(TfLiteType type) { return DataType(type); }
std::set<DeviceFlag> TfLiteUtil::GetAvailableDevices() const {
  static std::set<DeviceFlag> valid_devices = {};
  static std::once_flag once_flag;

  std::call_once(once_flag, [&]() {
    for (size_t flag = 0; flag < EnumLength<DeviceFlag>(); flag++) {
      const DeviceFlag device_flag = static_cast<DeviceFlag>(flag);
      if (TfLiteModelExecutor::GetDeviceDelegate(device_flag).ok()) {
        valid_devices.insert(device_flag);
      }
    }
  });

  return valid_devices;
}
}  // namespace tfl
}  // namespace band
