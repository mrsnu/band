#ifndef BAND_ENGINE_H_
#define BAND_ENGINE_H_

#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "band/c/common.h"
#include "band/common.h"
#include "band/config.h"
#include "band/context.h"
#include "band/error_reporter.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/tensor_ring_buffer.h"

namespace Band {

class Model;
class ModelSpec;
class LatencyEstimator;

typedef std::vector<Interface::ITensor*> Tensors;

/**
 * @brief The main entry point of the `Band`.
 * Public methods define an interface for
 * multi-DNN inference system.
 * Private methods inherit interface functions in Context class.
 *
 * Example usage:
 *
 * Band::RuntimeConfig config;
 * Band::ParseRuntimeConfigFromJson("band/test/data/config.json", config);
 * std::unique_ptr<Band::Engine> engine = Band::Engine::Create(config);
 *
 * Band::Model model;
 * model.FromPath(kBandTfLite, "band/test/data/add.bin");
 * engine->RegisterModel(&model);
 *
 * Band::Tensor *input_tensor = engine->CreateTensor(model.GetId(),
 * engine->GetInputTensorIndices(model.GetId())[0]);
 * Band::Tensor *output_tensor = engine->CreateTensor(model.GetId(),
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

  BandStatus RegisterModel(Model* model);
  BandStatus UnregisterModel(Model* model);

  Tensor* CreateTensor(ModelId model_id, int tensor_index);
  std::vector<int> GetOutputTensorIndices(ModelId model_id) const;
  std::vector<int> GetInputTensorIndices(ModelId model_id) const;

  size_t GetNumWorkers() const override;
  BandDeviceFlags GetWorkerDevice(WorkerId id) const;

  BandStatus RequestSync(
      ModelId model_id,
      BandRequestOption options = BandGetDefaultRequestOption(),
      Tensors inputs = {}, Tensors outputs = {});
  BandStatus RequestSync(std::vector<ModelId> model_ids,
                         std::vector<BandRequestOption> options = {},
                         std::vector<Tensors> inputs = {},
                         std::vector<Tensors> outputs = {});
  JobId RequestAsync(ModelId model_id,
                     BandRequestOption options = BandGetDefaultRequestOption(),
                     Tensors inputs = {});
  std::vector<JobId> RequestAsync(std::vector<ModelId> model_ids,
                                  std::vector<BandRequestOption> options = {},
                                  std::vector<Tensors> inputs = {});

  BandStatus Wait(JobId job_id, Tensors outputs = {});
  BandStatus Wait(std::vector<JobId> job_ids,
                  std::vector<Tensors> outputs = {});
  BandStatus GetOutputTensors(JobId job_id, Tensors outputs = {});

  // Sets the callback function pointer to report the end of invoke.
  void SetOnEndRequest(std::function<void(int, BandStatus)> on_end_request);

  int64_t GetProfiled(const SubgraphKey& key) const override;
  int64_t GetExpected(const SubgraphKey& key) const override;
  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override;

 private:
  /* context */
  BandStatus Init(const RuntimeConfig& config) override;
  WorkerWaitingTime GetWorkerWaitingTime() const override;
  std::set<WorkerId> GetIdleWorkers() const override;

  bool IsBegin(const SubgraphKey& key) const override;
  bool IsEnd(const SubgraphKey& key) const override;
  bool HasSubgraph(const SubgraphKey& key) const override;
  BandStatus Invoke(const SubgraphKey& key) override;

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
      Job& job,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;

  SubgraphKey GetSubgraphIdxSatisfyingSLO(
      Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
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
  /* getters */
  const Worker* GetWorker(WorkerId id) const override;
  Worker* GetWorker(WorkerId id) override;
  /* tensor communication */
  BandStatus TryCopyInputTensors(const Job& job) override;
  BandStatus TryCopyOutputTensors(const Job& job) override;

  /* helper functions */
  WorkerId GetDeviceWorkerId(BandDeviceFlags flag) const;
  Interface::IModelExecutor* GetModelExecutor(const SubgraphKey& key);
  const Interface::IModelExecutor* GetModelExecutor(
      const SubgraphKey& key) const;

  Engine() = delete;
  Engine(ErrorReporter* error_reporeter);
  Engine(const Engine&) = delete;
  Engine(const Engine&&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine& operator=(const Engine&&) = delete;

  ModelConfig model_config_;

  std::map<std::pair<ModelId, WorkerId>,
           std::unique_ptr<Interface::IModelExecutor>>
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
};  // namespace Band
}  // namespace Band

#endif  // BAND_ENGINE_H