#ifndef BAND_ENGINE_H_
#define BAND_ENGINE_H_

#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "band/common.h"
#include "band/config.h"
#include "band/engine_interface.h"
#include "band/error_reporter.h"
#include "band/estimator/estimator_interface.h"
#ifdef BAND_SPLASH
#include "band/estimator/thermal_estimator.h"
#else
#include "band/estimator/latency_estimator.h"
#endif  // BAND_SPLASH
#include "band/device/frequency.h"
#include "band/device/thermal.h"
#include "band/interface/model_executor.h"
#include "band/interface/tensor.h"
#include "band/profiler/latency_profiler.h"
#include "band/profiler/thermal_profiler.h"
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
  static std::unique_ptr<Engine> Create(
      const RuntimeConfig& config,
      ErrorReporter* error_reporter = DefaultErrorReporter());

  void SetDumpDirectory(std::string path);

  absl::Status RegisterModel(Model* model);
  absl::Status UnregisterModel(Model* model);

  Tensor* CreateTensor(ModelId model_id, int tensor_index);
  std::vector<int> GetOutputTensorIndices(ModelId model_id) const;
  std::vector<int> GetInputTensorIndices(ModelId model_id) const;

  size_t GetNumWorkers() const override;
  DeviceFlag GetWorkerDevice(WorkerId id) const override;

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

  absl::Status Wait(JobId job_id, Tensors outputs = {});
  absl::Status Wait(std::vector<JobId> job_ids,
                    std::vector<Tensors> outputs = {});
  void WaitAll();
  absl::Status GetOutputTensors(JobId job_id, Tensors outputs = {});

  // Sets the callback function pointer to report the end of invoke.
  void SetOnEndRequest(std::function<void(int, absl::Status)> on_end_request);

  double GetProfiled(const SubgraphKey& key) const override;
  double GetExpected(const SubgraphKey& key) const override;

  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override;

  Frequency* GetFrequency() const override { return frequency_.get(); }

  Thermal* GetThermal() const override {
    return thermal_profiler_->GetThermal();
  }

  void SetMinFrequencies() override {
    auto freq = GetFrequency();
    if (!freq->SetRuntimeFrequency(0.7104).ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set runtime frequeny");
    }
    if (!freq->SetCpuFrequency(0.8256).ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set cpu frequeny");
    }
    if (!freq->SetGpuFrequency(0.2570).ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set gpu frequeny");
    }
  }

  void SetMaxFrequencies() override {
    auto freq = GetFrequency();
    if (!freq->SetRuntimeFrequency(2.4192).ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set runtime frequeny");
    }
    if (!freq->SetCpuFrequency(2.8416).ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set cpu frequeny");
    }
    if (!freq->SetGpuFrequency(0.5850).ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to set gpu frequeny");
    }
  }

  void Sleep() override {
    SetMinFrequencies();
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * 60 * 1000));
    SetMaxFrequencies();
  }

 private:
  /* engine */
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
  std::pair<std::vector<SubgraphKey>, double> GetSubgraphWithMinCost(
      const Job& job, const WorkerWaitingTime& worker_waiting,
      const CostFunc cost) const override;

  std::pair<std::vector<SubgraphKey>, double> GetMinCostWithUnitSubgraph(
      ModelId model_id, int start_unit_idx,
      const WorkerWaitingTime& worker_waiting,
      const CostFunc cost) const override;

  SubgraphKey GetSubgraphIdxSatisfyingSLO(
      const Job& job, const WorkerWaitingTime& worker_waiting,
      const std::set<WorkerId>& idle_workers) const override;

  std::vector<SubgraphKey> GetSubgraphCandidates(
      ModelId model_id, BitMask resolved_unit_subgraphs) const;

  std::pair<SubgraphKey, double> GetMinCostSubgraphKey(
      const std::vector<SubgraphKey>& subgraph_keys, double start_time,
      ThermalMap start_them, FreqMap freq,
      const WorkerWaitingTime& worker_waiting, const CostFunc cost) const;

  /* estimators */
  void UpdateWithJob(const SubgraphKey&, Job& job) override;
  void Update(const SubgraphKey&, JobId job_id);

  /* planner */
  void Trigger() override;
  JobId EnqueueRequest(Job job, bool push_front = false) override;
  std::vector<JobId> EnqueueBatch(std::vector<Job> jobs,
                                  bool push_front = false) override;
  void PrepareReenqueue(Job& job) override;
  void EnqueueFinishedJob(Job& job) override;
  bool EnqueueToWorker(const ScheduleAction& schedule_action) override;
  bool EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) override;
  const Worker* GetWorker(WorkerId id) const override;
  Worker* GetWorker(WorkerId id) override;
  /* tensor communication */
  absl::Status TryCopyInputTensors(const Job& job) override;
  absl::Status TryCopyOutputTensors(const Job& job) override;

  /* helper functions */
  WorkerId GetDeviceWorkerId(DeviceFlag flag) const;
  interface::IModelExecutor* GetModelExecutor(const SubgraphKey& key);
  const interface::IModelExecutor* GetModelExecutor(
      const SubgraphKey& key) const;

  /* Model Profile */
  void BeginEvent(JobId job_id) override;
  void EndEvent(JobId job_id) override;
  absl::Status ProfileModel(ModelId model_id);

  Engine() = delete;
  Engine(ErrorReporter* error_reporeter);
  Engine(const Engine&) = delete;
  Engine(const Engine&&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine& operator=(const Engine&&) = delete;

  SubgraphConfig subgraph_config_;
  ProfileConfig profile_config_;

  std::map<std::pair<ModelId, WorkerId>,
           std::unique_ptr<interface::IModelExecutor>>
      model_executors_;
  mutable WorkerWaitingTime workers_waiting_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::unique_ptr<Planner> planner_;

  // Models
  // Maps to model spec
  std::map<ModelId, ModelSpec> model_specs_;
  std::map<ModelId, std::unique_ptr<TensorRingBuffer>> model_input_buffer_;
  std::map<ModelId, std::unique_ptr<TensorRingBuffer>> model_output_buffer_;

  std::string dump_dir_;

  // Scheduling

  // Find subgraph indices with the (model_id, start_unit_idx, end_unit_idx).
  // NOTE: we assume every subgraph consists of unit subgraphs with the
  // continuous unit subgraph indices.
  std::map<int, std::map<int, std::map<int, std::vector<SubgraphKey>>>>
      unit_subgraphs_to_subgraph_keys_;

  // Resources
  std::unique_ptr<Frequency> frequency_;
  std::unique_ptr<Thermal> thermal_;

  // Profilers
  std::unique_ptr<LatencyProfiler> latency_profiler_;
  std::unique_ptr<ThermalProfiler> thermal_profiler_;
  std::vector<Profiler*> profilers_;

  // Estimators
  std::unique_ptr<LatencyEstimator> latency_estimator_;
#ifdef BAND_SPLASH
  std::unique_ptr<ThermalEstimator> thermal_estimator_;
#endif  // BAND_SPLASH

  // Sleep engine with temperature target
  ThermalMap target_temp_;
};

}  // namespace band

#endif  // BAND_ENGINE_H