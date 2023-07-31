
#ifndef BAND_BACKEND_TFL_UTIL_H_
#define BAND_BACKEND_TFL_UTIL_H_

#include "absl/status/status.h"
#include "band/backend/tfl/tensorflow.h"
#include "band/common.h"
#include "band/interface/backend.h"


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