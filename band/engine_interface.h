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

#ifndef BAND_ENGINE_INTERFACE_H_
#define BAND_ENGINE_INTERFACE_H_

#include <functional>
#include <map>
#include <queue>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/logger.h"

namespace band {
namespace interface {
class IModel;
class IModelExecutor;
}  // namespace interface
class Worker;
class Planner;
class RuntimeConfig;
class Tensor;
class ModelSpec;

// Type definition for the device waiting time.
// The unit of time is ms.
using WorkerWaitingTime = std::map<WorkerId, int64_t>;

// Decision from a scheduler. Run subgraph key for a specific job.
using ScheduleAction = std::pair<Job, SubgraphKey>;

// Type definition of job queue.
using JobQueue = std::deque<Job>;

// Minimal interfaces for Band framework
class IEngine {
 public:
  IEngine() = default;
  virtual ~IEngine() = default;

  virtual absl::Status Init(const RuntimeConfig& config) {
    BAND_NOT_IMPLEMENTED;
    return absl::OkStatus();
  };

  /* worker */
  virtual void UpdateWorkersWaiting() const = 0;
  virtual WorkerWaitingTime GetWorkerWaitingTime() const = 0;
  virtual std::set<WorkerId> GetIdleWorkers() const = 0;

  /* subgraph */
  virtual SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                            WorkerId worker_id) const = 0;
  virtual bool IsBegin(const SubgraphKey& key) const = 0;
  virtual bool IsEnd(const SubgraphKey& key) const = 0;
  virtual bool HasSubgraph(const SubgraphKey& key) const = 0;
  virtual void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> visitor) const = 0;
  virtual absl::Status Invoke(const SubgraphKey& key) = 0;

  /* model */
  virtual const ModelSpec* GetModelSpec(ModelId model_id) const = 0;
  virtual WorkerId GetModelWorker(ModelId model_id) const = 0;

  /* scheduling */

  // Return a pair of the subgraph idx that leads to the shortest final
  // latency, and that final latency value.
  // Note that the returned subgraph may only cover a subset of the remaining
  // ops, but the latency value is calculated with all subgraphs leading to
  // the final op (of the model) in mind.

  // TODO: replace subgraph idx to subgraph key in below functions
  virtual std::pair<SubgraphKey, int64_t> GetShortestLatency(
      int model_id, BitMask resolved_unit_subgraphs, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;

  virtual std::pair<std::vector<SubgraphKey>, int64_t>
  GetShortestLatencyWithUnitSubgraph(
      int model_id, int start_unit_idx,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;

  virtual std::pair<std::vector<SubgraphKey>, int64_t>
  GetSubgraphWithShortestLatency(
      const Job& job,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;

  virtual SubgraphKey GetSubgraphIdxSatisfyingSLO(
      const Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
      const std::set<WorkerId>& idle_workers) const = 0;

  /* profiler */
  virtual void UpdateLatency(const SubgraphKey& key, int64_t latency) = 0;
  virtual int64_t GetProfiled(const SubgraphKey& key) const = 0;
  virtual int64_t GetExpected(const SubgraphKey& key) const = 0;

  /* planner */
  virtual void Trigger() = 0;
  virtual JobId EnqueueRequest(Job job, bool push_front = false) = 0;
  virtual std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                          bool push_front = false) = 0;
  virtual void PrepareReenqueue(Job& job) = 0;
  virtual void EnqueueFinishedJob(Job& job) = 0;
  virtual bool EnqueueToWorker(const ScheduleAction& schedule_action) = 0;
  virtual bool EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) = 0;

  /* getters */
  virtual const Worker* GetWorker(WorkerId id) const = 0;
  virtual Worker* GetWorker(WorkerId id) = 0;
  virtual size_t GetNumWorkers() const = 0;

  /* tensor communication */
  virtual absl::Status TryCopyInputTensors(const Job& job) = 0;
  virtual absl::Status TryCopyOutputTensors(const Job& job) = 0;
};
}  // namespace band

#endif  // BAND_ENGINE_INTERFACE_H_