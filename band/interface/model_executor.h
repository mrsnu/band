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

#ifndef BAND_INTERFACE_MODEL_EXECUTOR_H_
#define BAND_INTERFACE_MODEL_EXECUTOR_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"
#include "band/device/cpu.h"
#include "band/interface/backend.h"
#include "band/interface/model.h"
#include "band/model_spec.h"


namespace band {
namespace interface {
/*
  Model executor for specific <IModel, Worker>
*/

class ITensorView;
class IModelExecutor : public IBackendSpecific {
 public:
  IModelExecutor(
      ModelId model_id, WorkerId worker_id, DeviceFlag device_flag,
      CpuSet thread_affinity_mask = BandCPUMaskGetSet(CPUMaskFlag::kAll),
      int num_threads = -1)
      : model_id_(model_id),
        worker_id_(worker_id),
        device_flag_(device_flag),
        thread_affinity_mask_(thread_affinity_mask),
        num_threads_(num_threads > 0 ? num_threads : -1) {}
  virtual ~IModelExecutor() = default;

  virtual absl::StatusOr<ModelSpec> InvestigateModelSpec(IModel* model) = 0;
  virtual absl::Status PrepareSubgraph(IModel* model, std::set<int> ops = {},
                                       std::set<int> unit_indices = {}) = 0;

  virtual const std::vector<int>& GetInputs(const SubgraphKey& key) const = 0;
  virtual const std::vector<int>& GetOutputs(const SubgraphKey& key) const = 0;
  virtual const char* GetInputName(const SubgraphKey& key, int index) const = 0;
  virtual const char* GetOutputName(const SubgraphKey& key,
                                    int index) const = 0;
  virtual size_t GetNumTensors(const SubgraphKey& key) const = 0;
  virtual size_t GetNumNodes(const SubgraphKey& key) const = 0;

  virtual std::shared_ptr<ITensorView> GetTensorView(const SubgraphKey& key,
                                                     int index) = 0;

  virtual bool HasSubgraph(const SubgraphKey& key) const = 0;
  virtual SubgraphKey GetLargestSubgraphKey() const = 0;

  virtual absl::Status ExecuteSubgraph(const SubgraphKey& key) = 0;
  virtual void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> visitor) = 0;

 protected:
  const ModelId model_id_;
  const WorkerId worker_id_;
  const DeviceFlag device_flag_;
  const CpuSet thread_affinity_mask_;
  const int num_threads_;

 private:
  // Disable copy due to complexity
  IModelExecutor(const IModelExecutor&) = delete;
  IModelExecutor(const IModelExecutor&&) = delete;
  IModelExecutor& operator=(const IModelExecutor&) = delete;
  IModelExecutor& operator=(const IModelExecutor&&) = delete;
};
}  // namespace interface
}  // namespace band

#endif