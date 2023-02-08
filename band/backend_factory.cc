#include "band/backend_factory.h"

#include <mutex>

#include "band/logger.h"

namespace Band {
using namespace Interface;

#ifdef BAND_TFLITE
#ifdef _WIN32
extern bool TfLiteRegisterCreators();
#else
__attribute__((weak)) extern bool TfLiteRegisterCreators() { return false; }
#endif
#endif

// Expected process

void RegisterBackendInternal() {
  static std::once_flag g_flag;
  std::call_once(g_flag, [] {
#ifdef BAND_TFLITE
    if (TfLiteRegisterCreators()) {
      BAND_LOG_INTERNAL(BAND_LOG_INFO, "Register TFL backend");
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Failed to register TFL backend");
    }
#else
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "TFL backend is disabled.");
#endif
  });
}

std::map<BackendType, std::shared_ptr<Creator<IModelExecutor, ModelId,
                                                  WorkerId, DeviceFlags>>>
    BackendFactory::model_executor_creators_ = {};
std::map<BackendType, std::shared_ptr<Creator<IModel, ModelId>>>
    BackendFactory::model_creators_ = {};
std::map<BackendType, std::shared_ptr<Creator<IBackendUtil>>>
    BackendFactory::util_creators_ = {};

IModelExecutor* BackendFactory::CreateModelExecutor(
    BackendType backend, ModelId model_id, WorkerId worker_id,
    DeviceFlags device_flag) {
  RegisterBackendInternal();
  auto it = model_executor_creators_.find(backend);
  return it != model_executor_creators_.end()
             ? it->second->Create(model_id, worker_id, device_flag)
             : nullptr;
}

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
    Creator<IModelExecutor, ModelId, WorkerId, DeviceFlags>*
        model_executor_creator,
    Creator<IModel, ModelId>* model_creator,
    Creator<IBackendUtil>* util_creator) {
  model_executor_creators_[backend] = std::shared_ptr<
      Creator<IModelExecutor, ModelId, WorkerId, DeviceFlags>>(
      model_executor_creator);
  model_creators_[backend] =
      std::shared_ptr<Creator<IModel, ModelId>>(model_creator);
  util_creators_[backend] =
      std::shared_ptr<Creator<IBackendUtil>>(util_creator);
}
}  // namespace Band