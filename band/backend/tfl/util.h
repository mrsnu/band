
#ifndef BAND_BACKEND_TFL_UTIL_H_
#define BAND_BACKEND_TFL_UTIL_H_

#include "band/common.h"
#include "band/interface/backend.h"
#include "tensorflow/lite/c/common.h"

namespace Band {
namespace TfLite {
BandStatus GetBandStatus(TfLiteStatus status);
DataType GetBandType(TfLiteType type);

class TfLiteUtil : public Interface::IBackendUtil {
 public:
  std::set<BandDeviceFlags> GetAvailableDevices() const override;
};

}  // namespace TfLite
}  // namespace Band

#endif