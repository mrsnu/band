#ifndef BAND_ENGINE_INTERFACE_H_
#define BAND_ENGINE_INTERFACE_H_

#include <functional>
#include <map>
#include <queue>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/error_reporter.h"

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
using WorkerWaitingTime = std::map<WorkerId, double>;

// Decision from a scheduler. Run subgraph key for a specific job.
using ScheduleAction = std::pair<Job, SubgraphKey>;

// Minimal interfaces for Band framework
class IEngine {
 public:
  IEngine(ErrorReporter* error_reporeter = DefaultErrorReporter())
      : error_reporter_(error_reporeter) {}

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
      std::function<void(const SubgraphKey&)> iterator) const = 0;
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
  virtual std::pair<SubgraphKey, double> GetMinCost(
      int model_id, BitMask resolved_unit_subgraphs, double start_time,
      const WorkerWaitingTime& worker_waiting,
      const std::function<double(double, std::map<SensorFlag, double>)> cost)
      const = 0;

  virtual std::pair<std::vector<SubgraphKey>, double>
  GetMinCostWithUnitSubgraph(
      int model_id, int start_unit_idx, const WorkerWaitingTime& worker_waiting,
      const std::function<double(double, std::map<SensorFlag, double>)> cost)
      const = 0;

  virtual std::pair<std::vector<SubgraphKey>, double> GetSubgraphWithMinCost(
      const Job& job, const WorkerWaitingTime& worker_waiting,
      const std::function<double(double, std::map<SensorFlag, double>)> cost)
      const = 0;

  virtual SubgraphKey GetSubgraphIdxSatisfyingSLO(
      const Job& job, const WorkerWaitingTime& worker_waiting,
      const std::set<WorkerId>& idle_workers) const = 0;

  /* estimators */
  virtual void UpdateWithEvent(const SubgraphKey&, size_t event_id) = 0;
  virtual double GetProfiled(const SubgraphKey&) const = 0;
  virtual double GetExpected(const SubgraphKey&) const = 0;

  /* profilers */
  virtual size_t BeginEvent() = 0;
  virtual void EndEvent(size_t event_id) = 0;

  /* planner */
  virtual void Trigger() = 0;
  virtual JobId EnqueueRequest(Job job, bool push_front = false) = 0;
  virtual std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                          bool push_front = false) = 0;
  virtual void PrepareReenqueue(Job& job) = 0;
  virtual void EnqueueFinishedJob(Job& job) = 0;
  virtual bool EnqueueToWorker(const ScheduleAction& schedule_action,
                               const int idle_us = -1) = 0;
  virtual bool EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action,
      const std::vector<int> idle_uses = {}) = 0;

  /* getters */
  virtual const ErrorReporter* GetErrorReporter() const {
    return error_reporter_;
  }
  virtual const Worker* GetWorker(WorkerId id) const = 0;
  virtual Worker* GetWorker(WorkerId id) = 0;
  virtual size_t GetNumWorkers() const = 0;
  virtual DeviceFlag GetWorkerDevice(WorkerId id) const = 0;

  /* tensor communication */
  virtual absl::Status TryCopyInputTensors(const Job& job) = 0;
  virtual absl::Status TryCopyOutputTensors(const Job& job) = 0;

 protected:
  ErrorReporter* error_reporter_;
};
}  // namespace band

#endif  // BAND_ENGINE_INTERFACE_H_