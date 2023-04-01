#ifndef BAND_ENGINE_H_
#define BAND_ENGINE_H_

#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "band/common.h"
#include "band/config.h"
#include "band/context.h"
#include "band/error_reporter.h"
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
 *
 * Example usage:
 *
 * band::RuntimeConfig config;
 * band::ParseRuntimeConfigFromJson("band/test/data/config.json", config);
 * std::unique_ptr<band::Engine> engine = band::Engine::Create(config);
 *
 * Band::Model model;
 * model.FromPath(BackendType::TfLite, "band/test/data/add.tflite");
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
class Engine : public Context {
 public:
  ~Engine() override;
  static std::unique_ptr<Engine> Create(
      const RuntimeConfig& config,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  absl::Status RegisterModel(Model* model);
  absl::Status UnregisterModel(Model* model);

  Tensor* CreateTensor(ModelId model_id, int tensor_index);
  std::vector<int> GetOutputTensorIndices(ModelId model_id) const;
  std::vector<int> GetInputTensorIndices(ModelId model_id) const;

  size_t GetNumWorkers() const override;
  DeviceFlags GetWorkerDevice(WorkerId id) const;

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
      std::vector<ModelId> model_ids,
      std::vector<RequestOption> options = {},
      std::vector<Tensors> inputs = {});

  absl::Status Wait(JobId job_id, Tensors outputs = {});
  absl::Status Wait(std::vector<JobId> job_ids,
                    std::vector<Tensors> outputs = {});
  void WaitAll();
  absl::Status GetOutputTensors(JobId job_id, Tensors outputs = {});

  // Sets the callback function pointer to report the end of invoke.
  void SetOnEndRequest(std::function<void(int, absl::Status)> on_end_request);

  int64_t GetProfiled(const SubgraphKey& key) const override;
  int64_t GetExpected(const SubgraphKey& key) const override;
  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override;

 private:
  /* context */
  absl::Status Init(const RuntimeConfig& config) override;
  void UpdateWorkersWaiting() const override;
  WorkerWaitingTime GetWorkerWaitingTime() const override;
  std::set<WorkerId> GetIdleWorkers() const override;

  bool IsBegin(const SubgraphKey& key) const override;
  bool IsEnd(const SubgraphKey& key) const override;
  bool HasSubgraph(const SubgraphKey& key) const override;
  void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> iterator) const override;
  absl::Status Invoke(const SubgraphKey& key) override;

  const ModelSpec* GetModelSpec(ModelId model_id) const override;
  WorkerId GetModelWorker(ModelId model_id) const override;

  /* utility funtions for unit-level scheduling */
  std::pair<SubgraphKey, int64_t> GetShortestLatency(
      ModelId model_id, BitMask resolved_unit_subgraphs, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;

  std::pair<std::vector<SubgraphKey>, int64_t>
  GetShortestLatencyWithUnitSubgraph(
      ModelId model_id, int start_unit_idx,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;

  std::pair<std::vector<SubgraphKey>, int64_t> GetSubgraphWithShortestLatency(
      const Job& job,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;

  SubgraphKey GetSubgraphIdxSatisfyingSLO(
      const Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
      const std::set<WorkerId>& idle_workers) const override;

  std::vector<SubgraphKey> GetSubgraphCandidates(
      ModelId model_id, BitMask resolved_unit_subgraphs) const;

  std::pair<SubgraphKey, int64_t> GetShortestSubgraphKey(
      const std::vector<SubgraphKey>& subgraph_keys, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting) const;

  /* latency estimator */
  void UpdateLatency(const SubgraphKey& key, int64_t latency) override;
  int64_t GetWorst(ModelId model_id) const;

  /* planner */
  void Trigger() override;
  JobId EnqueueRequest(Job job, bool push_front = false) override;
  std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                  bool push_front = false) override;
  void PrepareReenqueue(Job& job) override;
  void EnqueueFinishedJob(Job& job) override;
  void EnqueueToWorker(const ScheduleAction& schedule_action) override;
  void EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) override;
  const Worker* GetWorker(WorkerId id) const override;
  Worker* GetWorker(WorkerId id) override;
  /* tensor communication */
  absl::Status TryCopyInputTensors(const Job& job) override;
  absl::Status TryCopyOutputTensors(const Job& job) override;

  /* helper functions */
  WorkerId GetDeviceWorkerId(DeviceFlags flag) const;
  interface::IModelExecutor* GetModelExecutor(const SubgraphKey& key);
  const interface::IModelExecutor* GetModelExecutor(
      const SubgraphKey& key) const;

  Engine() = delete;
  Engine(ErrorReporter* error_reporeter);
  Engine(const Engine&) = delete;
  Engine(const Engine&&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine& operator=(const Engine&&) = delete;

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
                             std::pair<SubgraphKey, int64_t>, CacheHash>
      cache_;

  // Find subgraph indices with the (model_id, start_unit_idx, end_unit_idx).
  // NOTE: we assume every subgraph consists of unit subgraphs with the
  // continuous unit subgraph indices.
  std::map<int, std::map<int, std::map<int, std::vector<SubgraphKey>>>>
      unit_subgraphs_to_subgraph_keys_;
};  // namespace band
}  // namespace band

#endif  // BAND_ENGINE_H