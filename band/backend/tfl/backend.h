/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
                                            DeviceFlag, CpuSet, int> {
 public:
  IModelExecutor* Create(ModelId model_id, WorkerId worker_id,
                         DeviceFlag device_flag, CpuSet thread_affinity_mask,
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