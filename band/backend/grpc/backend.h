#ifndef BAND_BACKEND_GRPC_BACKEND_H_
#define BAND_BACKEND_GRPC_BACKEND_H_

#include "band/backend/grpc/model.h"
#include "band/backend/grpc/model_executor.h"
#include "band/backend/grpc/tensor.h"
#include "band/backend/grpc/util.h"
#include "band/backend_factory.h"
#include "band/interface/backend.h"

namespace band {

using namespace interface;
namespace grpc {

class ModelExecutorCreator
    : public Creator<IModelExecutor, ModelId, WorkerId, DeviceFlag,
                     std::shared_ptr<BackendConfig>, CpuSet, int> {
 public:
  IModelExecutor* Create(ModelId model_id, WorkerId worker_id,
                         DeviceFlag device_flag,
                         std::shared_ptr<BackendConfig> config,
                         CpuSet thread_affinity_mask,
                         int num_threads) const override {
    return new GrpcModelExecutor(model_id, worker_id, device_flag, config,
                                 thread_affinity_mask, num_threads);
  }
};

class ModelCreator : public Creator<IModel, ModelId> {
 public:
  IModel* Create(ModelId id) const override { return new GrpcModel(id); }
};

class UtilCreator : public Creator<IBackendUtil> {
 public:
  IBackendUtil* Create() const override { return new GrpcUtil(); }
};

}  // namespace grpc
}  // namespace band

#endif  // BAND_BACKEND_GRPC_BACKEND_H_