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
#endif  // _WIN32
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
#ifdef _WIN32
extern bool GrpcRegisterCreators();
#else
__attribute__((weak)) extern bool GrpcRegisterCreators() { return false; }
#endif  // _WIN32
#endif  // BAND_GRPC

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
#endif  // BAND_TFLITE

#ifdef BAND_GRPC
    if (GrpcRegisterCreators()) {
      BAND_LOG_INTERNAL(BAND_LOG_INFO, "Register GRPC backend");
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_ERROR, "Failed to register GRPC backend");
    }
#else
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "GRPC backend is disabled.");
#endif  // BAND_GRPC
  });
}

std::map<BackendType, std::shared_ptr<Creator<
                          IModelExecutor, ModelId, WorkerId, DeviceFlags,
                          std::shared_ptr<BackendConfig>, CpuSet, int>>>
    BackendFactory::model_executor_creators_ = {};
std::map<BackendType, std::shared_ptr<Creator<IModel, ModelId>>>
    BackendFactory::model_creators_ = {};
std::map<BackendType, std::shared_ptr<Creator<IBackendUtil>>>
    BackendFactory::util_creators_ = {};

IModelExecutor* BackendFactory::CreateModelExecutor(
    BackendType backend, ModelId model_id, WorkerId worker_id,
    DeviceFlags device_flag,
    std::shared_ptr<BackendConfig> backend_config,
    CpuSet thread_affinity_mask, int num_threads) {
  RegisterBackendInternal();
  auto it = model_executor_creators_.find(backend);
  return it != model_executor_creators_.end()
             ? it->second->Create(model_id, worker_id, device_flag,
                                  backend_config, thread_affinity_mask,
                                  num_threads)
             : nullptr;
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
    Creator<IModelExecutor, ModelId, WorkerId, DeviceFlags,
            std::shared_ptr<BackendConfig>, CpuSet, int>*
        model_executor_creator,
    Creator<IModel, ModelId>* model_creator,
    Creator<IBackendUtil>* util_creator) {
  model_executor_creators_[backend] = std::shared_ptr<
      Creator<IModelExecutor, ModelId, WorkerId, DeviceFlags,
              std::shared_ptr<BackendConfig>, CpuSet, int>>(
      model_executor_creator);
  model_creators_[backend] =
      std::shared_ptr<Creator<IModel, ModelId>>(model_creator);
  util_creators_[backend] =
      std::shared_ptr<Creator<IBackendUtil>>(util_creator);
}
}  // namespace band