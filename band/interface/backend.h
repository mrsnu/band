#ifndef BAND_INTERFACE_BACKEND_H_
#define BAND_INTERFACE_BACKEND_H_

namespace band {
namespace interface {
class IBackendSpecific {
  // 定义后端相关的特定功能或属性。它主要用于识别和比较后端类型
 public:
  virtual BackendType GetBackendType() const = 0;
  // BackendType是一个枚举或类似的类型，用于区分不同的后端（如TensorFlow、PyTorch等）
  bool IsCompatible(const IBackendSpecific& rhs) const {
    return IsCompatible(&rhs);
  }
  bool IsCompatible(const IBackendSpecific* rhs) const {
    return GetBackendType() == rhs->GetBackendType();
  }
  // 两个重载的方法提供了一种检查两个后端实例是否兼容（即是否为相同类型的后端）的机制
  // 通过比较两个实例的BackendType来实现。
};

class IBackendUtil {
  // 定义了一组用于查询后端支持的设备和其他实用功能的接口。
  // 这个接口使得应用程序可以查询后端提供的设备信息，如可用的CPU、GPU等。
 public:
  virtual std::set<DeviceFlag> GetAvailableDevices() const = 0;
};

}  // namespace interface
}  // namespace band

#endif  // BAND_INTERFACE_BACKEND_H_