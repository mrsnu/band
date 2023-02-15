
#ifndef BAND_BACKEND_TFL_UTIL_H_
#define BAND_BACKEND_TFL_UTIL_H_

#include "band/common.h"
#include "band/interface/backend.h"
#include "tensorflow/lite/c/common.h"

namespace band {
namespace tfl {
BandStatus GetBandStatus(TfLiteStatus status);
BandType GetBandType(TfLiteType type);

class TfLiteUtil : public interface::IBackendUtil {
 public:
  std::set<BandDeviceFlags> GetAvailableDevices() const override;
};

}  // namespace tfl
}  // namespace band

#endif