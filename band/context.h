#ifndef BAND_CONTEXT_H_
#define BAND_CONTEXT_H_

#include <map>
#include <queue>
#include <unordered_map>

#include "band/c/common.h"
#include "band/common.h"
#include "band/config.h"
#include "band/error_reporter.h"

namespace Band {
namespace Interface {
class IModel;
class IInterpreter;
}  // namespace Interface
class Worker;
class Profiler;
class Planner;
class RuntimeConfig;
class Tensor;

// Type definition for the device waiting time.
// The unit of time is ms.
using WorkerWaitingTime = std::map<WorkerId, int64_t>;

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
  virtual void UpdateWorkerWaitingTime() const = 0;
  virtual const WorkerWaitingTime& GetWorkerWaitingTime() const = 0;
  virtual std::set<WorkerId> GetIdleWorkers() const = 0;

  /* subgraph */
  virtual SubgraphKey GetModelSubgraphKey(ModelId model_id,
                                          WorkerId worker_id) const = 0;
  virtual bool IsEnd(const SubgraphKey& key) const = 0;
  virtual BandStatus Invoke(const SubgraphKey& key) = 0;

  /* model */
  virtual const ModelSpec* GetModelSpec(ModelId model_id) = 0;
  virtual int GetModelConfigIdx(ModelId model_id) const = 0;
  virtual WorkerId GetModelWorker(ModelId model_id) const = 0;

  /* scheduling */

  // Return a pair of the subgraph idx that leads to the shortest final
  // latency, and that final latency value.
  // Note that the returned subgraph may only cover a subset of the remaining
  // ops, but the latency value is calculated with all subgraphs leading to
  // the final op (of the model) in mind.

  // TODO: replace subgraph idx to subgraph key in below functions
  virtual std::pair<SubgraphKey, int64_t> GetShortestLatency(
      int model_id, std::set<int> resolved_tensors, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting,
      SubgraphKey preceded_subgraph_index = {}) const = 0;

  virtual std::pair<std::vector<SubgraphKey>, int64_t>
  GetShortestLatencyWithUnitSubgraph(
      int model_id, int start_unit_idx,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;

  virtual std::pair<std::vector<SubgraphKey>, int64_t>
  GetSubgraphWithShortestLatency(
      Job& job, const std::map<WorkerId, int64_t>& worker_waiting) const = 0;

  virtual SubgraphKey GetSubgraphIdxSatisfyingSLO(
      Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
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

  /* getters */
  virtual ErrorReporter* GetErrorReporter() { return error_reporter_; }
  virtual Worker* GetWorker(WorkerId id) = 0;

  /* tensor communication */
  virtual BandStatus TryCopyInputTensors(const Job& job) = 0;
  virtual BandStatus TryCopyOutputTensors(const Job& job) = 0;

 protected:
  ErrorReporter* error_reporter_;
};
}  // namespace Band

#endif  // BAND_CONTEXT_H_