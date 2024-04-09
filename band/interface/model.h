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
  // 构造函数，初始化id_成员变量
  virtual ~IModel() = default;
  // 析构函数，使用默认实现

  virtual absl::Status FromPath(const char* filename) = 0;
  // 从文件路径加载模型
  virtual absl::Status FromBuffer(const char* buffer, size_t buffer_size) = 0;
  // 从缓冲区加载模型，允许直接从内存中加载模型
  virtual bool IsInitialized() const = 0;
  // 检查模型是否已经初始化
  ModelId GetId() const { return id_; }
  // 获取模型的id
  const std::string& GetPath() const { return path_; }
  // 获取模型的路径

 protected:
  std::string path_;
  const ModelId id_;
};
}  // namespace interface
}  // namespace band

#endif