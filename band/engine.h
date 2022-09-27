#ifndef BAND_ENGINE_H

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
#include "band/tensor_ring_buffer.h"

namespace Band {
namespace Interface {
class IInterpreter;
class ITensor;
}  // namespace Interface
class Model;

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
 * Band::ParseRuntimeConfigFromJson("band/testdata/config.json", config);
 * std::unique_ptr<Band::Engine> engine = Band::Engine::Create(config);
 *
 * Band::Model model;
 * model.FromPath(kBandTfLite, "band/testdata/add.bin");
 * engine->RegisterModel(&model);
 *
 * Band::Tensor *input_tensor = engine->CreateTensor(model.GetId(),
 * engine->GetInputTensorIndices(model.GetId())[0]);
 * Band::Tensor *output_tensor = engine->CreateTensor(model.GetId(),
 * engine->GetOutputTensorIndices(model.GetId())[0]);
 *
 * // Copy input data to input_tensor->GetData()
 * engine->InvokeSyncModel(model.GetId(), {input_tensor}, {output_tensor})
 * // Copy result from output_tensor->GetData()
 */
class Engine : public Context {
 public:
  ~Engine() override;
  static std::unique_ptr<Engine> Create(
      const RuntimeConfig& config,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  BandStatus RegisterModel(Model* model);
  Tensor* CreateTensor(ModelId model_id, int tensor_index);
  std::vector<int> GetOutputTensorIndices(ModelId model_id) const;
  std::vector<int> GetInputTensorIndices(ModelId model_id) const;

  BandStatus InvokeSyncModel(ModelId model_id, Tensors inputs = {},
                             Tensors outputs = {});
  BandStatus InvokeSyncModels(std::vector<ModelId> model_ids,
                              std::vector<Tensors> inputs = {},
                              std::vector<Tensors> outputs = {});
  JobId InvokeAsyncModel(ModelId model_id, Tensors inputs = {});
  std::vector<JobId> InvokeAsyncModels(std::vector<ModelId> model_ids,
                                       std::vector<Tensors> inputs = {});
  BandStatus Wait(JobId job_id, Tensors outputs = {});
  BandStatus Wait(std::vector<JobId> job_ids,
                  std::vector<Tensors> outputs = {});
  BandStatus GetOutputTensors(JobId job_id, Tensors outputs = {});

  // Sets the callback function pointer to report the end of invoke.
  void SetEndInvokeFunction(std::function<void(int, BandStatus)> on_end_invoke);

 private:
  /* context */
  BandStatus Init(const RuntimeConfig& config) override;
  void UpdateWorkerWaitingTime() const override;
  const WorkerWaitingTime& GetWorkerWaitingTime() const override;
  std::set<WorkerId> GetIdleWorkers() const override;
  SubgraphKey GetModelSubgraphKey(ModelId model_id,
                                  WorkerId worker_id) const override;
  bool IsEnd(const SubgraphKey& key) const override;
  BandStatus Invoke(const SubgraphKey& key) override;
  ModelSpec* GetModelSpec(ModelId model_id) { return &model_specs_[model_id]; }

  std::pair<SubgraphKey, int64_t> GetShortestLatency(
      int model_id, std::set<int> resolved_tensors, int64_t start_time,
      const std::map<WorkerId, int64_t>& worker_waiting,
      SubgraphKey preceded_subgraph_index = {}) const override;

  std::pair<std::vector<SubgraphKey>, int64_t>
  GetShortestLatencyWithUnitSubgraph(
      int model_id, int start_unit_idx,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;

  std::pair<std::vector<SubgraphKey>, int64_t> GetSubgraphWithShortestLatency(
      Job& job,
      const std::map<WorkerId, int64_t>& worker_waiting) const override;

  SubgraphKey GetSubgraphIdxSatisfyingSLO(
      Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
      const std::set<WorkerId>& idle_workers) const override;
  /* profiler */
  void UpdateLatency(const SubgraphKey& key, int64_t latency) override;
  int64_t GetProfiled(const SubgraphKey& key) const override;
  int64_t GetExpected(const SubgraphKey& key) const override;
  /* planner */
  void Trigger() override;
  JobId EnqueueRequest(Job job, bool push_front = false) override;
  std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                  bool push_front = false) override;
  void PrepareReenqueue(Job& job) override;
  void EnqueueFinishedJob(Job& job) override;
  /* getters */
  Worker* GetWorker(WorkerId id) override;
  /* tensor communication */
  BandStatus TryCopyInputTensors(const Job& job) override;
  BandStatus TryCopyOutputTensors(const Job& job) override;

  /* helper functions */
  WorkerId GetDeviceWorkerId(BandDeviceFlags flag) const;
  Interface::IInterpreter* GetInterpreter(const SubgraphKey& key);
  const Interface::IInterpreter* GetInterpreter(const SubgraphKey& key) const;

  Engine() = delete;
  Engine(ErrorReporter* error_reporeter);
  Engine(const Engine&) = delete;
  Engine(const Engine&&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine& operator=(const Engine&&) = delete;

  struct ModelOption {
    // Minimum subgraph size.
    // Will not create subgraph if num operators < minimum_subgraph_size.
    int minimum_subgraph_size_;

    // Subgraph preparation type
    // "fallback_per_device", "unit_subgraph"
    std::string subgraph_preparation_type_;
  } model_option_;

  std::map<std::pair<WorkerId, ModelId>,
           std::unique_ptr<Interface::IInterpreter>>
      interpreters_;
  std::map<WorkerId, std::unique_ptr<Worker>> workers_;
  mutable WorkerWaitingTime workers_waiting_;
  std::unique_ptr<Profiler> profiler_;
  std::unique_ptr<Planner> planner_;

  // Models
  // Maps to each model's configuration.
  std::map<ModelId, ModelConfig> model_configs_;
  // Maps to model spec
  std::map<ModelId, ModelSpec> model_specs_;

  std::map<ModelId, std::unique_ptr<TensorRingBuffer>> model_input_buffer_;
  std::map<ModelId, std::unique_ptr<TensorRingBuffer>> model_output_buffer_;

  // Scheduling
  // cache for GetShortestLatency()
  std::unordered_map<std::pair<int, std::set<int>>, std::pair<int, int64_t>,
                     PairHash>
      cache_;
  // Find subgraph indices with the (model_id, start_unit_idx, end_unit_idx).
  // NOTE: we assume every subgraph consists of unit subgraphs with the
  // continuous unit subgraph indices.
  std::map<int, std::map<int, std::map<int, std::vector<int>>>>
      unit_subgraphs_to_subgraph_indices_;
};  // namespace Band
}  // namespace Band

#endif  // BAND_ENGINE_H