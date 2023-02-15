#ifndef BAND_CONTEXT_H_
#define BAND_CONTEXT_H_

#include <functional>
#include <map>
#include <queue>
#include <unordered_map>

#include "band/c/common.h"
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
using WorkerWaitingTime = std::map<WorkerId, int64_t>;

// Decision from a scheduler. Run subgraph key for a specific job.
using ScheduleAction = std::pair<Job, SubgraphKey>;

// Type definition of job queue.
using JobQueue = std::deque<Job>;

// Minimal interfaces for Band framework
class Context {
 public:
  Context(ErrorReporter* error_reporeter = DefaultErrorReporter())
      : error_reporter_(error_reporeter) {}

  virtual ~Context() = default;

  virtual BandStatus Init(const RuntimeConfig& config) {
    BAND_NOT_IMPLEMENTED;
    return kBandOk;
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
  virtual BandStatus Invoke(const SubgraphKey& key) = 0;

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
  virtual void EnqueueToWorker(const ScheduleAction& schedule_action) = 0;
  virtual void EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) = 0;

  /* getters */
  virtual const ErrorReporter* GetErrorReporter() const {
    return error_reporter_;
  }
  virtual const Worker* GetWorker(WorkerId id) const = 0;
  virtual Worker* GetWorker(WorkerId id) = 0;
  virtual size_t GetNumWorkers() const = 0;

  /* tensor communication */
  virtual BandStatus TryCopyInputTensors(const Job& job) = 0;
  virtual BandStatus TryCopyOutputTensors(const Job& job) = 0;

 protected:
  ErrorReporter* error_reporter_;
};
}  // namespace band

#endif  // BAND_CONTEXT_H_