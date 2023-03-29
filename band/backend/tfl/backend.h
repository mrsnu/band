#ifndef BAND_BACKEND_TFL_BACKEND_H
#define BAND_BACKEND_TFL_BACKEND_H

#include "band/backend/tfl/model.h"
#include "band/backend/tfl/model_executor.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend/tfl/util.h"
#include "band/backend_factory.h"
#include "band/interface/backend.h"

namespace band {
using namespace interface;
namespace tfl {
class ModelExecutorCreator : public Creator<IModelExecutor, ModelId, WorkerId,
                                            BandDeviceFlags, CpuSet, int> {
 public:
  IModelExecutor* Create(ModelId model_id, WorkerId worker_id,
                         BandDeviceFlags device_flag,
                         CpuSet thread_affinity_mask,
                         int num_threads) const override {
    return new TfLiteModelExecutor(model_id, worker_id, device_flag,
                                   thread_affinity_mask, num_threads);
  }
};

class ModelCreator : public Creator<IModel, ModelId> {
 public:
  IModel* Create(ModelId id) const override { return new TfLiteModel(id); }
};

class UtilCreator : public Creator<IBackendUtil> {
 public:
  IBackendUtil* Create() const override { return new TfLiteUtil(); }
};

}  // namespace tfl
}  // namespace band

#endif