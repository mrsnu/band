#ifndef BAND_BACKEND_FACTORY_H_
#define BAND_BACKEND_FACTORY_H_

#include "band/backend_factory.h"
#include "band/common.h"
#include "band/interface/model.h"
#include "band/interface/model_executor.h"

// Expected workflow (per each backend)
// 1. Implement creators
// 2. Register them with `RegisterBackendCreators` in arbitrary global function
// (e.g., tfl::RegisterCreators)
// 3. Add an arbitrary global function to `RegisterBackendInternal` with
// corresponding flag

namespace band {

template <typename Base, class... Args>
struct Creator {
 public:
  virtual Base* Create(Args...) const { return nullptr; };
};

class BackendFactory {
 public:
  static interface::IModelExecutor* CreateModelExecutor(
      BackendType backend, ModelId model_id, WorkerId worker_id,
      DeviceFlag device_flag,
      CpuSet thread_affinity_mask = BandCPUMaskGetSet(CPUMaskFlag::kAll),
      int num_threads = -1);
  static interface::IModel* CreateModel(BackendType backend, ModelId id);
  static interface::IBackendUtil* GetBackendUtil(BackendType backend);
  static std::vector<BackendType> GetAvailableBackends();

  static void RegisterBackendCreators(
      BackendType backend,
      Creator<interface::IModelExecutor, ModelId, WorkerId, DeviceFlag, CpuSet,
              int>* model_executor_creator,
      Creator<interface::IModel, ModelId>* model_creator,
      Creator<interface::IBackendUtil>* util_creator);

 private:
  BackendFactory() = default;

  static std::map<BackendType,
                  std::shared_ptr<Creator<interface::IModelExecutor, ModelId,
                                          WorkerId, DeviceFlag, CpuSet, int>>>
      model_executor_creators_;
  static std::map<BackendType,
                  std::shared_ptr<Creator<interface::IModel, ModelId>>>
      model_creators_;
  static std::map<BackendType,
                  std::shared_ptr<Creator<interface::IBackendUtil>>>
      util_creators_;
};
}  // namespace band

#endif