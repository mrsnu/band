
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