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
//   更新工作等待时间
  virtual WorkerWaitingTime GetWorkerWaitingTime() const = 0;
//   获取工作等待时间
  virtual std::set<WorkerId> GetIdleWorkers() const = 0;
//   获取空闲工作

  /* subgraph */
  virtual SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                            WorkerId worker_id) const = 0;
                                            // 获取最大子图键
  virtual bool IsBegin(const SubgraphKey& key) const = 0;
//   判断是否是开始
  virtual bool IsEnd(const SubgraphKey& key) const = 0;
  virtual bool HasSubgraph(const SubgraphKey& key) const = 0;
  virtual void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> visitor) const = 0;
    //   是否遍历处理每一个子图
  virtual absl::Status Invoke(const SubgraphKey& key) = 0;
//   调用子图

  /* model */
  virtual const ModelSpec* GetModelSpec(ModelId model_id) const = 0;
//   获取模型规范
  virtual WorkerId GetModelWorker(ModelId model_id) const = 0;
//   获取执行指定模型的工作器id

  /* scheduling */

  // Return a pair of the subgraph idx that leads to the shortest final
  // latency, and that final latency value.
  // Note that the returned subgraph may only cover a subset of the remaining
  // ops, but the latency value is calculated with all subgraphs leading to
  // the final op (of the model) in mind.
//   返回一个元组，包括导致最短最终延迟的子图索引及该延迟值。
// 需要注意的是，虽然返回的子图可能仅包含剩余操作中的一部分，但计算延迟值时已综合考虑了所有导向模型最后一个操作的子图。


  // TODO: replace subgraph idx to subgraph key in below functions
  // 待办事项：在后续函数中，计划将子图索引替换为子图键。
  virtual std::pair<SubgraphKey, int64_t> GetShortestLatency(
      int model_id, BitMask resolved_unit_subgraphs, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;
//   获取最短延迟

  virtual std::pair<std::vector<SubgraphKey>, int64_t>
  GetShortestLatencyWithUnitSubgraph(
      int model_id, int start_unit_idx,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;
//   获取最短延迟与单元子图

  virtual std::pair<std::vector<SubgraphKey>, int64_t>
  GetSubgraphWithShortestLatency(
      const Job& job,
      const std::map<WorkerId, int64_t>& worker_waiting) const = 0;
//   获取最短延迟的子图

  virtual SubgraphKey GetSubgraphIdxSatisfyingSLO(
      const Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
      const std::set<WorkerId>& idle_workers) const = 0;
//   获取满足SLO的子图索引

  /* profiler */
  virtual void UpdateLatency(const SubgraphKey& key, int64_t latency) = 0;
//   更新延迟
  virtual int64_t GetProfiled(const SubgraphKey& key) const = 0;
//   获取已经分析的
  virtual int64_t GetExpected(const SubgraphKey& key) const = 0;
//   获取预期的

  /* planner */
  virtual void Trigger() = 0;
//   触发
  virtual JobId EnqueueRequest(Job job, bool push_front = false) = 0;
//   入队请求
  virtual std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                          bool push_front = false) = 0;
//   批量入队
  virtual void PrepareReenqueue(Job& job) = 0;
//   准备重新入队
  virtual void EnqueueFinishedJob(Job& job) = 0;
//   入队完成的工作
  virtual bool EnqueueToWorker(const ScheduleAction& schedule_action) = 0;
//   入队到工作
  virtual bool EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) = 0;
//   批量入队到工作

  /* getters */
  virtual const Worker* GetWorker(WorkerId id) const = 0;
//   获取工作
  virtual Worker* GetWorker(WorkerId id) = 0;
//   获取工作
  virtual size_t GetNumWorkers() const = 0;
//   获取工作数量

  /* tensor communication */
  virtual absl::Status TryCopyInputTensors(const Job& job) = 0;
//   尝试复制输入张量
  virtual absl::Status TryCopyOutputTensors(const Job& job) = 0;
//   尝试复制输出张量
};
}  // namespace band

#endif  // BAND_ENGINE_INTERFACE_H_