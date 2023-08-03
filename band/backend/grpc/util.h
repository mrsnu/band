#ifndef BAND_BACKEND_GRPC_UTIL_H_
#define BAND_BACKEND_GRPC_UTIL_H_

#include "band/interface/backend.h"

namespace band {
namespace grpc {

class GrpcUtil : public interface::IBackendUtil {
 public:
  std::set<DeviceFlag> GetAvailableDevices() const override {
    return {DeviceFlag::kNETWORK};
  }
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_UTIL_H_