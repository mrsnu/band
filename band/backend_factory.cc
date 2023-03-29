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

std::map<BandBackendType,
         std::shared_ptr<Creator<IModelExecutor, ModelId, WorkerId,
                                 BandDeviceFlags, CpuSet, int>>>
    BackendFactory::model_executor_creators_ = {};
std::map<BandBackendType, std::shared_ptr<Creator<IModel, ModelId>>>
    BackendFactory::model_creators_ = {};
std::map<BandBackendType, std::shared_ptr<Creator<IBackendUtil>>>
    BackendFactory::util_creators_ = {};

IModelExecutor* BackendFactory::CreateModelExecutor(
    BandBackendType backend, ModelId model_id, WorkerId worker_id,
    BandDeviceFlags device_flag, CpuSet thread_affinity_mask, int num_threads) {
  RegisterBackendInternal();
  auto it = model_executor_creators_.find(backend);
  return it != model_executor_creators_.end()
             ? it->second->Create(model_id, worker_id, device_flag,
                                  thread_affinity_mask, num_threads)
             : nullptr;
}

IModel* BackendFactory::CreateModel(BandBackendType backend, ModelId id) {
  RegisterBackendInternal();
  auto it = model_creators_.find(backend);
  return it != model_creators_.end() ? it->second->Create(id) : nullptr;
}

IBackendUtil* BackendFactory::GetBackendUtil(BandBackendType backend) {
  RegisterBackendInternal();
  auto it = util_creators_.find(backend);
  return it != util_creators_.end() ? it->second->Create() : nullptr;
}

std::vector<BandBackendType> BackendFactory::GetAvailableBackends() {
  RegisterBackendInternal();
  // assume static creators are all valid - after instantiation to reach here
  std::vector<BandBackendType> valid_backends;

  for (auto type_model_executor_creator : model_executor_creators_) {
    valid_backends.push_back(type_model_executor_creator.first);
  }

  return valid_backends;
}

void BackendFactory::RegisterBackendCreators(
    BandBackendType backend,
    Creator<IModelExecutor, ModelId, WorkerId, BandDeviceFlags, CpuSet, int>*
        model_executor_creator,
    Creator<IModel, ModelId>* model_creator,
    Creator<IBackendUtil>* util_creator) {
  model_executor_creators_[backend] = std::shared_ptr<
      Creator<IModelExecutor, ModelId, WorkerId, BandDeviceFlags, CpuSet, int>>(
      model_executor_creator);
  model_creators_[backend] =
      std::shared_ptr<Creator<IModel, ModelId>>(model_creator);
  util_creators_[backend] =
      std::shared_ptr<Creator<IBackendUtil>>(util_creator);
}
}  // namespace band