#ifndef BAND_ENGINE_H_
#define BAND_ENGINE_H_

#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "band/common.h"
#include "band/config.h"
#include "band/engine_interface.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/tensor_ring_buffer.h"

namespace band {

class Model;
class ModelSpec;
class LatencyEstimator;

typedef std::vector<interface::ITensor*> Tensors;

/**
 * @brief The main entry point of the `Band`.
 * Public methods define an interface for
 * multi-DNN inference system.
 * Private methods inherit interface functions in Context class.
 * @brief `Band`的主要入口点。
 * 公开的方法为多-DNN推理系统定义了一套接口。
 * 私有方法从 Context 类继承接口函数。
 *
 * Example usage:
 *
 * band::RuntimeConfig config;
 * band::ParseRuntimeConfigFromJson("band/test/data/config.json", config);
 * std::unique_ptr<band::Engine> engine = band::Engine::Create(config);
 *
 * Band::Model model;
 * model.FromPath(BackendType::kTfLite, "band/test/data/add.tflite");
 * engine->RegisterModel(&model);
 *
 * band::Tensor *input_tensor = engine->CreateTensor(model.GetId(),
 * engine->GetInputTensorIndices(model.GetId())[0]);
 * band::Tensor *output_tensor = engine->CreateTensor(model.GetId(),
 * engine->GetOutputTensorIndices(model.GetId())[0]);
 *
 * // Copy input data to input_tensor->GetData()
 * engine->RequestSync(model.GetId(), {input_tensor}, {output_tensor})
 * // Copy result from output_tensor->GetData()
 */
class Engine : public IEngine {
 public:
  ~Engine() override;
  static std::unique_ptr<Engine> Create(const RuntimeConfig& config);
//   构造和析构函数，静态方法，根据提供的config创建一个Engine实例 返回一个unique_ptr

  absl::Status RegisterModel(Model* model);
  absl::Status UnregisterModel(Model* model);
// 用于注册和注销模型 保证enign能够追踪哪些模型当前是激活的

  Tensor* CreateTensor(ModelId model_id, int tensor_index);
//   根据模型ID和张量索引创建张量，用于模型计算中的数据存储和处理
  std::vector<int> GetOutputTensorIndices(ModelId model_id) const;
  std::vector<int> GetInputTensorIndices(ModelId model_id) const;
//   获取模型的输入和输出张量索引，有助于在执行模型时的管理数据流

  size_t GetNumWorkers() const override;
  DeviceFlag GetWorkerDevice(WorkerId id) const;
// 工作器管理 获取工作起的数量和设备标志

  absl::Status RequestSync(
      ModelId model_id,
      RequestOption options = RequestOption::GetDefaultOption(),
      Tensors inputs = {}, Tensors outputs = {});
  absl::Status RequestSync(std::vector<ModelId> model_ids,
                           std::vector<RequestOption> options = {},
                           std::vector<Tensors> inputs = {},
                           std::vector<Tensors> outputs = {});

  absl::StatusOr<JobId> RequestAsync(
      ModelId model_id,
      RequestOption options = RequestOption::GetDefaultOption(),
      Tensors inputs = {});
  absl::StatusOr<std::vector<JobId>> RequestAsync(
      std::vector<ModelId> model_ids, std::vector<RequestOption> options = {},
      std::vector<Tensors> inputs = {});
// 处理同步和异步执行的模型请求，支持单个或者多个模型的执行
// 异步执行返回JobId，用于后续的查询和处理

  absl::Status Wait(JobId job_id, Tensors outputs = {});
  absl::Status Wait(std::vector<JobId> job_ids,
                    std::vector<Tensors> outputs = {});
  void WaitAll();
  absl::Status GetOutputTensors(JobId job_id, Tensors outputs = {});
//   等待模型执行完成，获取输出张量

  // Sets the callback function pointer to report the end of invoke.
//   设置回调函数指针以报告调用结束
  CallbackId SetOnEndRequest(
      std::function<void(int, absl::Status)> on_end_request);
  absl::Status UnsetOnEndRequest(CallbackId callback_id);
//   设置和取消设置请求结束时的回调函数，这有助于在作业完成时进行通知或进一步处理

  int64_t GetProfiled(const SubgraphKey& key) const override;
  int64_t GetExpected(const SubgraphKey& key) const override;
//   获取特定子图的性能分析数据和预期执行时间，用于优化和调度决策。

  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override;
                                    // 获取在特定工作器上的模型中最大子图的键，这可能用于资源优化和加载平衡

 private:
  /* engine */
  absl::Status Init(const RuntimeConfig& config) override;
//   初始化引擎 配置运行时参数
  void UpdateWorkersWaiting() const override;
//   更新工作器的等待时间
  WorkerWaitingTime GetWorkerWaitingTime() const override;
//   获取工作器的等待时间
  std::set<WorkerId> GetIdleWorkers() const override;
//   获取空闲工作器

  bool IsBegin(const SubgraphKey& key) const override;
  bool IsEnd(const SubgraphKey& key) const override;
  bool HasSubgraph(const SubgraphKey& key) const override;
  void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> visitor) const override;
    //   遍历所有子图，为每个子图执行提供的访问函数
  absl::Status Invoke(const SubgraphKey& key) override;
//   执行特定子图

  const ModelSpec* GetModelSpec(ModelId model_id) const override;
//   获取模型规范
  WorkerId GetModelWorker(ModelId model_id) const override;
//   获取模型的工作器

  /* utility funtions for unit-level scheduling */
//   单元级调度的实用功能
  std::pair<SubgraphKey, int64_t> GetShortestLatency(
      ModelId model_id, BitMask resolved_unit_subgraphs, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;
    //   获取最短延迟

  std::pair<std::vector<SubgraphKey>, int64_t>
  GetShortestLatencyWithUnitSubgraph(
      ModelId model_id, int start_unit_idx,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;
//   获取最短延迟和单元子图

  std::pair<std::vector<SubgraphKey>, int64_t> GetSubgraphWithShortestLatency(
      const Job& job,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;
//   获取最短延迟的子图

  SubgraphKey GetSubgraphIdxSatisfyingSLO(
      const Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
      const std::set<WorkerId>& idle_workers) const override;
//   获取满足SLO的子图

  std::vector<SubgraphKey> GetSubgraphCandidates(
      ModelId model_id, BitMask resolved_unit_subgraphs) const;
//   获取子图候选

  std::pair<SubgraphKey, int64_t> GetShortestSubgraphKey(
      const std::vector<SubgraphKey>& subgraph_keys, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting) const;
//   获取最短子图键

  /* latency estimator */
//   延迟估计器
  void UpdateLatency(const SubgraphKey& key, int64_t latency) override;
  int64_t GetWorst(ModelId model_id) const;
//   更新延迟和获取最差延迟

  /* planner */
//   计划器
  void Trigger() override;
  JobId EnqueueRequest(Job job, bool push_front = false) override;
  std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                  bool push_front = false) override;
                                //   触发计划器，将作业加入队列
  void PrepareReenqueue(Job& job) override;
//   准备重新入队
  void EnqueueFinishedJob(Job& job) override;
//   将完成的作业加入队列
  bool EnqueueToWorker(const ScheduleAction& schedule_action) override;
//   将作业加入工作器
  bool EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) override;
    //   将作业批量加入工作器
  const Worker* GetWorker(WorkerId id) const override;
//   获取工作器
  Worker* GetWorker(WorkerId id) override;
//   获取工作器

  /* tensor communication */
//   张量通信
  absl::Status TryCopyInputTensors(const Job& job) override;
  absl::Status TryCopyOutputTensors(const Job& job) override;

  /* helper functions */
//   辅助函数
  WorkerId GetDeviceWorkerId(DeviceFlag flag) const;
  interface::IModelExecutor* GetModelExecutor(const SubgraphKey& key);
  const interface::IModelExecutor* GetModelExecutor(
      const SubgraphKey& key) const;

  Engine() = default;
  Engine(const Engine&) = delete;
  Engine(const Engine&&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine& operator=(const Engine&&) = delete;
//   禁用复制和移动构造函数及赋值操作符

  SubgraphConfig subgraph_config_;

  std::map<std::pair<ModelId, WorkerId>,
           std::unique_ptr<interface::IModelExecutor>>
      model_executors_;
  std::vector<std::unique_ptr<Worker>> workers_;
  mutable WorkerWaitingTime workers_waiting_;
  std::unique_ptr<LatencyEstimator> latency_estimator_;
  std::unique_ptr<Planner> planner_;

  // Models

  // Maps to model spec
  std::map<ModelId, ModelSpec> model_specs_;
  std::map<ModelId, std::unique_ptr<TensorRingBuffer>> model_input_buffer_;
  std::map<ModelId, std::unique_ptr<TensorRingBuffer>> model_output_buffer_;

  // Scheduling
  // cache for GetShortestLatency()
  mutable std::unordered_map<std::pair<ModelId, BitMask>,
                             std::pair<SubgraphKey, int64_t>, JobIdBitMaskHash>
      cache_;

  // Find subgraph indices with the (model_id, start_unit_idx, end_unit_idx).
  // NOTE: we assume every subgraph consists of unit subgraphs with the
  // continuous unit subgraph indices.
  std::map<int, std::map<int, std::map<int, std::vector<SubgraphKey>>>>
      unit_subgraphs_to_subgraph_keys_;
};  // namespace band
}  // namespace band

#endif  // BAND_ENGINE_H