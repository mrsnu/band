#include "band/backend_factory.h"

#include <mutex>

#include "band/logger.h"

namespace band {
using namespace interface;

#ifdef BAND_TFLITE
#ifdef _WIN32
extern bool TfLiteRegisterCreators();
#else
__attribute__((weak)) extern bool TfLiteRegisterCreators() { return false; }
#endif
#endif

// Expected process
void RegisterBackendInternal() {
  // 负责注册所有可用的后端，并且它确保注册过程只执行一次，无论这个函数被调用多少次。
  static std::once_flag g_flag;
  std::call_once(g_flag, [] {
#ifdef BAND_TFLITE
    if (TfLiteRegisterCreators()) {
      BAND_LOG(LogSeverity::kInfo, "Register TFL backend");
    } else {
      BAND_LOG(LogSeverity::kError, "Failed to register TFL backend");
    }
#else
    BAND_LOG(LogSeverity::kInfo, "TFL backend is disabled.");
#endif
  });
}

std::map<BackendType, std::shared_ptr<Creator<IModelExecutor, ModelId, WorkerId,
                                              DeviceFlag, CpuSet, int>>>
    BackendFactory::model_executor_creators_ = {};
std::map<BackendType, std::shared_ptr<Creator<IModel, ModelId>>>
    BackendFactory::model_creators_ = {};
std::map<BackendType, std::shared_ptr<Creator<IBackendUtil>>>
    BackendFactory::util_creators_ = {};
// 初始了三个成员变量，用于将BackendType（后端类型）映射到相应的创建器对象。
// 这些创建器对象是用于动态创建模型执行器（IModelExecutor）、模型（IModel）和后端工具（IBackendUtil）的工厂对象。

IModelExecutor* BackendFactory::CreateModelExecutor(
    BackendType backend, ModelId model_id, WorkerId worker_id,
    DeviceFlag device_flag, CpuSet thread_affinity_mask, int num_threads) {
  RegisterBackendInternal();
  auto it = model_executor_creators_.find(backend);
  // 查找给定backend类型的条目
  return it != model_executor_creators_.end()
             ? it->second->Create(model_id, worker_id, device_flag,
                                  thread_affinity_mask, num_threads)
             : nullptr;
            //  如果找到了对应的创建器对象（it != model_executor_creators_.end()），
            // 则调用该创建器的Create方法，并传递给定的参数，以创建并返回一个新的模型执行器实例。
}  // namespace band

IModel* BackendFactory::CreateModel(BackendType backend, ModelId id) {
  RegisterBackendInternal();
  auto it = model_creators_.find(backend);
  return it != model_creators_.end() ? it->second->Create(id) : nullptr;
}

IBackendUtil* BackendFactory::GetBackendUtil(BackendType backend) {
  RegisterBackendInternal();
  auto it = util_creators_.find(backend);
  return it != util_creators_.end() ? it->second->Create() : nullptr;
}

std::vector<BackendType> BackendFactory::GetAvailableBackends() {
  RegisterBackendInternal();
  // assume static creators are all valid - after instantiation to reach here
  std::vector<BackendType> valid_backends;

  for (auto type_model_executor_creator : model_executor_creators_) {
    valid_backends.push_back(type_model_executor_creator.first);
  }

  return valid_backends;
}

void BackendFactory::RegisterBackendCreators(
    BackendType backend,
    // 指定要为其注册创建器的后端类型
    Creator<IModelExecutor, ModelId, WorkerId, DeviceFlag, CpuSet, int>*
        model_executor_creator,
    Creator<IModel, ModelId>* model_creator,
    Creator<IBackendUtil>* util_creator) {
  model_executor_creators_[backend] = std::shared_ptr<
      Creator<IModelExecutor, ModelId, WorkerId, DeviceFlag, CpuSet, int>>(
      model_executor_creator);
  model_creators_[backend] =
      std::shared_ptr<Creator<IModel, ModelId>>(model_creator);
  util_creators_[backend] =
      std::shared_ptr<Creator<IBackendUtil>>(util_creator);
  // 使用std::shared_ptr进行封装主要是为了自动管理创建器对象的生命周期，避免内存泄漏，并允许在多处共享这些创建器对象。
}
}  // namespace band