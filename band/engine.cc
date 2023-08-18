#include "band/engine.h"

#include <algorithm>
#include <cassert>

#include "absl/strings/str_format.h"
#include "band/backend_factory.h"
#include "band/common.h"
#include "band/engine_interface.h"
#include "band/estimator/latency_estimator.h"
#include "band/estimator/thermal_estimator.h"
#include "band/interface/tensor_view.h"
#include "band/job_tracer.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/model_analyzer.h"
#include "band/model_spec.h"
#include "band/planner.h"
#include "band/profiler/frequency_profiler.h"
#include "band/profiler/latency_profiler.h"
#include "band/profiler/thermal_profiler.h"
#include "band/tensor.h"
#include "band/worker.h"

namespace band {

Engine::~Engine() {
  for (auto& model_executor : model_executors_) {
    model_executor.second.reset();
  }

  for (auto& worker : workers_) {
    worker->End();
  }

  for (auto& worker : workers_) {
    worker.reset();
  }

  planner_.reset();

  delete latency_profiler_;
  delete thermal_profiler_;
  delete frequency_profiler_;
}

std::unique_ptr<Engine> Engine::Create(const RuntimeConfig& config,
                                       ErrorReporter* error_reporter) {
  std::unique_ptr<Engine> engine_ptr(new Engine(error_reporter));
  return engine_ptr->Init(config).ok() ? std::move(engine_ptr) : nullptr;
}

absl::Status Engine::RegisterModel(Model* model) {
  if (!model) {
    return absl::InternalError("Model is empty.");
  }

  if (model->GetSupportedBackends().size() == 0) {
    return absl::InternalError("No supported backends.");
  }

  const ModelId model_id = model->GetId();

  for (BackendType backend_type : model->GetSupportedBackends()) {
    // Analyze model & generate subgraphs per backend type
    ModelAnalyzer analyzer(*this, planner_->NeedFallbackSubgraphs(),
                           subgraph_config_, model, backend_type);

    const auto status_or_result = analyzer.CreateSubgraphs();
    if (!status_or_result.ok()) {
      // TODO(BAND-49): unregister for specific backend
      auto status = UnregisterModel(model);
      if (!status.ok()) {
        BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to unregister model %d: %s",
                      model_id, status.message());
      }
      return status_or_result.status();
    }

    const auto result = analyzer.CreateSubgraphs().value();
    const ModelSpec model_spec = std::get<0>(result);
    const std::vector<SubgraphDef> subgraph_defs = std::get<1>(result);

    // Create internal model_executor per each supported backends
    {
      bool added_once = false;
      for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
        if (model_spec.unavailable_devices.find(GetWorkerDevice(worker_id)) ==
            model_spec.unavailable_devices.end()) {
          const Worker* worker = workers_[worker_id].get();
          std::unique_ptr<interface::IModelExecutor> model_executor(
              BackendFactory::CreateModelExecutor(
                  backend_type, model_id, worker_id, GetWorkerDevice(worker_id),
                  worker->GetWorkerThreadAffinity(), worker->GetNumThreads()));
          model_executors_[{model_id, worker_id}] = std::move(model_executor);
          added_once = true;
          BAND_LOG_INTERNAL(BAND_LOG_INFO,
                            "Create model executor for model %d worker %s",
                            model_id, ToString(GetWorkerDevice(worker_id)));
        }
      }

      if (!added_once) {
        // TODO(BAND-49): unregister for specific backend
        auto status = UnregisterModel(model);
        if (!status.ok()) {
          BAND_LOG_PROD(BAND_LOG_ERROR, "Failed to unregister model %d: %s",
                        model_id, status.message());
        }
        return absl::InternalError(
            "Failed to create model executor on all worker types");
      }
    }

    model_specs_.insert({model_id, model_spec});

    // Prepare execution of subgraph definitions per each model_executor
    {
      for (const SubgraphDef& subgraph_def : subgraph_defs) {
        const std::pair<ModelId, WorkerId> model_executor_key = {
            model_id, subgraph_def.worker_id};
        const SubgraphKey key = {model_id, subgraph_def.worker_id,
                                 subgraph_def.unit_subgraph_indices};

        if (model_executors_.find(model_executor_key) ==
            model_executors_.end()) {
          BAND_REPORT_ERROR(error_reporter_,
                            "Subgraph logic created a subgraph for worker %d "
                            "that does not supports model %d",
                            subgraph_def.worker_id, model_id);
        } else {
          auto& model_executor =
              model_executors_[{model_id, subgraph_def.worker_id}];
          absl::Status status = model_executor->PrepareSubgraph(
              model->GetBackendModel(backend_type), subgraph_def.op_indices,
              subgraph_def.unit_subgraph_indices);
          if (status.ok()) {
            // Verify generated subgraphs
            if (model_executor->HasSubgraph(key) == false) {
              return absl::InternalError(absl::StrFormat(
                  "A subgraph for worker %d that does not exists",
                  subgraph_def.worker_id));
            }
            const std::set<int> inputs =
                model_spec.GetPureInputTensors(subgraph_def.op_indices);
            const std::set<int> all_outputs =
                model_spec.GetOutputTensors(subgraph_def.op_indices);

            if (!std::equal(model_executor->GetInputs(key).begin(),
                            model_executor->GetInputs(key).end(),
                            inputs.begin())) {
              return absl::InternalError(
                  absl::StrFormat("Input format is not correct for worker %d",
                                  subgraph_def.worker_id));
            }
            if (!std::includes(all_outputs.begin(), all_outputs.end(),
                               model_executor->GetOutputs(key).begin(),
                               model_executor->GetOutputs(key).end())) {
              return absl::InternalError(
                  absl::StrFormat("Output format is not correct for worker %d",
                                  subgraph_def.worker_id));
            }

            unit_subgraphs_to_subgraph_keys_
                [model_id][*subgraph_def.unit_subgraph_indices.begin()]
                [*subgraph_def.unit_subgraph_indices.rbegin()]
                    .push_back(key);
          }
        }
      }

      // Verify equality of all tensor pairs
      for (const SubgraphDef& lhs : subgraph_defs) {
        auto& lhs_model_executor = model_executors_[{model_id, lhs.worker_id}];
        const SubgraphKey lhs_key = {model_id, lhs.worker_id,
                                     lhs.unit_subgraph_indices};

        std::set<int> lhs_outputs{
            lhs_model_executor->GetOutputs(lhs_key).begin(),
            lhs_model_executor->GetOutputs(lhs_key).end()};

        for (const SubgraphDef& rhs : subgraph_defs) {
          auto& rhs_model_executor =
              model_executors_[{model_id, rhs.worker_id}];
          const SubgraphKey rhs_key = {model_id, rhs.worker_id,
                                       rhs.unit_subgraph_indices};
          if ((lhs.worker_id != rhs.worker_id) && (&lhs != &rhs)) {
            std::set<int> rhs_inputs{
                rhs_model_executor->GetInputs(rhs_key).begin(),
                rhs_model_executor->GetInputs(rhs_key).end()};

            std::set<int> common_tensors;
            std::set_intersection(
                lhs_outputs.begin(), lhs_outputs.end(), rhs_inputs.begin(),
                rhs_inputs.end(),
                std::inserter(common_tensors, common_tensors.end()));

            for (int common_tensor_index : common_tensors) {
              if (!(*lhs_model_executor->GetTensorView(lhs_key,
                                                       common_tensor_index) ==
                    *rhs_model_executor->GetTensorView(rhs_key,
                                                       common_tensor_index))) {
                return absl::InternalError(absl::StrFormat(
                    "%s %s %d != %s %s %d",
                    ToString(GetWorkerDevice(lhs.worker_id)),
                    lhs.ToString().c_str(), common_tensor_index,
                    ToString(GetWorkerDevice(rhs.worker_id)),
                    rhs.ToString().c_str(), common_tensor_index));
              }
            }
          }
        }
      }

      // todo: connect prev / next && unit indices

      // Initialize tensor ring buffer
      // Assumption: each backend model in band::Model has the same input /
      // output tensor shapes
      {
        std::vector<std::shared_ptr<interface::ITensor>> input_tensors;
        std::vector<std::shared_ptr<interface::ITensor>> output_tensors;

        auto model_subgraph_key = GetLargestSubgraphKey(
            model_id, GetDeviceWorkerId(DeviceFlag::kCPU));
        interface::IModelExecutor* primary_model_executor =
            GetModelExecutor(model_subgraph_key);

        for (int input_tensor : model_spec.input_tensors) {
          input_tensors.push_back(primary_model_executor->GetTensorView(
              model_subgraph_key, input_tensor));
        }

        for (int output_tensor : model_spec.output_tensors) {
          output_tensors.push_back(primary_model_executor->GetTensorView(
              model_subgraph_key, output_tensor));
        }

        const std::vector<int> input_indices{model_spec.input_tensors.begin(),
                                             model_spec.input_tensors.end()};
        const std::vector<int> output_indices{model_spec.output_tensors.begin(),
                                              model_spec.output_tensors.end()};

        model_input_buffer_.emplace(
            model->GetId(), std::make_unique<TensorRingBuffer>(
                                error_reporter_, input_tensors, input_indices));
        model_output_buffer_.emplace(
            model_id, std::make_unique<TensorRingBuffer>(
                          error_reporter_, output_tensors, output_indices));
      }
    }

    // Profile models
    {
      auto status = ProfileModel(model_id);
      if (!status.ok()) {
        return status;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status Engine::UnregisterModel(Model* model) {
  if (!model) {
    return absl::InternalError("Failed to unregister null model.");
  }

  for (auto it = model_executors_.begin(); it != model_executors_.end();) {
    (it->first.second == model->GetId()) ? model_executors_.erase(it++)
                                         : (++it);
  }

  for (auto it = model_specs_.begin(); it != model_specs_.end();) {
    (it->first == model->GetId()) ? model_specs_.erase(it++) : (++it);
  }

  for (auto it = model_input_buffer_.begin();
       it != model_input_buffer_.end();) {
    (it->first == model->GetId()) ? model_input_buffer_.erase(it++) : (++it);
  }

  for (auto it = model_output_buffer_.begin();
       it != model_output_buffer_.end();) {
    (it->first == model->GetId()) ? model_output_buffer_.erase(it++) : (++it);
  }

  return absl::OkStatus();
}

Tensor* Engine::CreateTensor(ModelId model_id, int tensor_index) {
  // TODO: What if there are multiple backends?
  SubgraphKey model_subgraph_key =
      GetLargestSubgraphKey(model_id, GetDeviceWorkerId(DeviceFlag::kCPU));

  if (interface::IModelExecutor* model_executor =
          GetModelExecutor(model_subgraph_key)) {
    return new Tensor(
        model_executor->GetTensorView(model_subgraph_key, tensor_index).get());
  } else {
    return nullptr;
  }
}

std::vector<int> Engine::GetOutputTensorIndices(ModelId model_id) const {
  SubgraphKey model_subgraph_key =
      GetLargestSubgraphKey(model_id, GetDeviceWorkerId(DeviceFlag::kCPU));
  const interface::IModelExecutor* model_executor =
      GetModelExecutor(model_subgraph_key);
  return model_executor ? model_executor->GetOutputs(model_subgraph_key)
                        : std::vector<int>();
}

std::vector<int> Engine::GetInputTensorIndices(ModelId model_id) const {
  SubgraphKey model_subgraph_key =
      GetLargestSubgraphKey(model_id, GetDeviceWorkerId(DeviceFlag::kCPU));
  const interface::IModelExecutor* model_executor =
      GetModelExecutor(model_subgraph_key);
  return model_executor ? model_executor->GetInputs(model_subgraph_key)
                        : std::vector<int>();
}

size_t Engine::GetNumWorkers() const { return workers_.size(); }

DeviceFlag Engine::GetWorkerDevice(WorkerId id) const {
  if (id >= 0 && id < workers_.size()) {
    return workers_.at(id)->GetDeviceFlag();
  }
  BAND_LOG_PROD(
      BAND_LOG_ERROR,
      "Cannot find the device for the given worker: %d. Fallback to CPU", id);
  return DeviceFlag::kCPU;
}

absl::Status Engine::RequestSync(ModelId model_id, RequestOption options,
                                 Tensors inputs, Tensors outputs) {
  auto status_or_job_id = RequestAsync(model_id, options, inputs);
  if (!status_or_job_id.ok()) {
    return status_or_job_id.status();
  }
  return Wait(status_or_job_id.value(), outputs);
}

absl::Status Engine::RequestSync(std::vector<ModelId> model_ids,
                                 std::vector<RequestOption> options,
                                 std::vector<Tensors> inputs,
                                 std::vector<Tensors> outputs) {
  auto status_or_job_ids = RequestAsync(model_ids, options, inputs);
  if (!status_or_job_ids.ok()) {
    return status_or_job_ids.status();
  }
  return Wait(status_or_job_ids.value(), outputs);
}

absl::StatusOr<JobId> Engine::RequestAsync(ModelId model_id,
                                           RequestOption options,
                                           Tensors inputs) {
  std::vector<Tensors> input_tensors;
  if (inputs.size()) {
    input_tensors.push_back(inputs);
  }
  auto status_or_job_ids = RequestAsync({model_id}, {options}, input_tensors);
  if (!status_or_job_ids.ok()) {
    return status_or_job_ids.status();
  }
  return status_or_job_ids.value()[0];
}

absl::StatusOr<std::vector<JobId>> Engine::RequestAsync(
    std::vector<ModelId> model_ids, std::vector<RequestOption> options,
    std::vector<Tensors> inputs) {
  std::vector<Job> jobs;

  if (model_ids.size() != options.size()) {
    return absl::InternalError(
        absl::StrFormat("# Model requests (%llu) != # Worker ids (%llu)",
                        model_ids.size(), options.size()));
  }

  for (size_t i = 0; i < model_ids.size(); i++) {
    // TODO(BAND-33): explicit job life cycle
    Job job(model_ids[i]);
    job.require_callback = options[i].require_callback;

    int target_slo_us = options[i].slo_us;
    // TODO(widiba03304): absl::optional for implicit slo_scale default.
    if (options[i].slo_scale != -1) {
      if (options[i].slo_scale <= 0) {
        return absl::InternalError(absl::StrFormat(
            "Specified slo_scale is invalid (%f <= 0)", options[i].slo_scale));
      }

      target_slo_us = GetWorst(model_ids[i]) * options[i].slo_scale;
    }

    // override, if `slo_us` is specified
    if (options[i].slo_us != -1) {
      target_slo_us = options[i].slo_us;
    }

    job.slo_us = target_slo_us;

    if (options[i].target_worker != -1) {
      Worker* target_worker = GetWorker(options[i].target_worker);
      if (target_worker == nullptr) {
        return absl::InternalError(
            absl::StrFormat("Request assigned to invalid worker id (%d)",
                            options[i].target_worker));
      }
      job.target_worker_id = options[i].target_worker;
    }

    if (i < inputs.size()) {
      int input_handle = model_input_buffer_[model_ids[i]]->Alloc();
      if (!model_input_buffer_[model_ids[i]]
               ->PutTensorsToHandle(inputs[i], input_handle)
               .ok()) {
        return absl::InternalError(
            absl::StrFormat("Input copy failure for model %d", model_ids[i]));
      }
      job.input_handle = input_handle;
      job.output_handle = model_output_buffer_[model_ids[i]]->Alloc();
    }

    jobs.push_back(job);
  }
  return EnqueueBatch(jobs);
}

absl::Status Engine::Wait(JobId job_id, Tensors outputs) {
  std::vector<Tensors> output_tensors;
  if (outputs.size()) {
    output_tensors.push_back(outputs);
  }
  return Wait(std::vector<JobId>({job_id}), output_tensors);
}

absl::Status Engine::Wait(std::vector<JobId> job_ids,
                          std::vector<Tensors> outputs) {
  for (auto job_id : job_ids) {
    BAND_LOG_PROD(BAND_LOG_INFO, "Wait for job %d", job_id);
  }

  planner_->Wait(job_ids);
  for (size_t i = 0; i < outputs.size(); i++) {
    auto status = GetOutputTensors(job_ids[i], outputs[i]);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

void Engine::WaitAll() { planner_->WaitAll(); }

absl::Status Engine::GetOutputTensors(JobId job_id, Tensors outputs) {
  Job job = planner_->GetFinishedJob(job_id);

  if (outputs.empty() || job_id == -1) {
    return absl::InternalError(
        absl::StrFormat("Invalid job id / num outputs to copy: (%d, %d)",
                        job_id, outputs.size()));
  }

  // Not finished or invalidated
  if (job.job_id == -1) {
    return absl::InternalError("Invalid job id / not finished or invalidated.");
  }

  if (job.output_handle == -1) {
    return absl::InternalError(
        absl::StrFormat("Invalid output handle : %d", job.output_handle));
  }

  if (job.status == JobStatus::kSLOViolation) {
    return absl::DeadlineExceededError("SLO violation");
  } else if (job.status != JobStatus::kSuccess) {
    return absl::InternalError(
        absl::StrFormat("Job failed with status : %s", ToString(job.status)));
  }

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    return absl::InternalError(
        absl::StrFormat("Invalid model id : %d", job.model_id));
  }

  auto status = model_output_buffer_.at(job.model_id)
                    ->GetTensorsFromHandle(outputs, job.output_handle);
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

void Engine::SetOnEndRequest(
    std::function<void(int, absl::Status)> on_end_request) {
  planner_->SetOnEndRequest(on_end_request);
}

absl::Status Engine::Init(const RuntimeConfig& config) {
  planner_ = std::make_unique<Planner>(*this);
  auto status = planner_->Init(config.planner_config);
  if (!status.ok()) {
    return status;
  }

  BAND_LOG_PROD(BAND_LOG_INFO, "MinimumSubgraphSize: %d",
                config.subgraph_config.minimum_subgraph_size);
  BAND_LOG_PROD(BAND_LOG_INFO, "PreparationType: %s",
                ToString(config.subgraph_config.subgraph_preparation_type));

  subgraph_config_ = config.subgraph_config;

  // Setup for profilers
  {
    latency_profiler_ = new LatencyProfiler();
    thermal_profiler_ = new ThermalProfiler(config.device_config);
    frequency_profiler_ = new FrequencyProfiler(config.device_config);
    profilers_ = {latency_profiler_, thermal_profiler_, frequency_profiler_};
  }

  // Setup for estimators
  {
    {
      latency_estimator_ =
          std::make_unique<LatencyEstimator>(this, latency_profiler_);
      auto status =
          latency_estimator_->Init(config.profile_config.latency_config);
      if (!status.ok()) {
        return status;
      }
    }
    {
      thermal_estimator_ =
          std::make_unique<ThermalEstimator>(this, thermal_profiler_);
      auto status =
          thermal_estimator_->Init(config.profile_config.thermal_config);
      if (!status.ok()) {
        return status;
      }
    }
    {
      frequency_latency_estimator_ =
          std::make_unique<FrequencyLatencyEstimator>(this, frequency_profiler_,
                                                      latency_profiler_);
      auto status = frequency_latency_estimator_->Init(
          config.profile_config.frequency_latency_config);
      if (!status.ok()) {
        return status;
      }
    }
  }

#if BAND_IS_MOBILE
  {
    const CPUMaskFlag cpu_mask = static_cast<CPUMaskFlag>(config.cpu_mask);
    auto cpu_mask_set = BandCPUMaskGetSet(cpu_mask);

    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Set affinity to %s cores.",
                      ToString(cpu_mask));

    auto status = SetCPUThreadAffinity(cpu_mask_set);
    if (!status.ok()) {
      return status;
    }
  }
#endif

  // Search for all available backends, devices
  std::set<DeviceFlag> valid_devices;
  auto valid_backends = BackendFactory::GetAvailableBackends();
  for (auto backend : valid_backends) {
    auto backend_devices =
        BackendFactory::GetBackendUtil(backend)->GetAvailableDevices();
    valid_devices.insert(backend_devices.begin(), backend_devices.end());
  }

  auto& potential_workers = config.worker_config.workers;
  for (int i = 0; i < potential_workers.size(); i++) {
    DeviceFlag device_flag = potential_workers[i];
    if (valid_devices.find(device_flag) != valid_devices.end()) {
      std::unique_ptr<Worker> worker;
      if (planner_->GetWorkerType() ==
          static_cast<int>(WorkerType::kGlobalQueue)) {
        worker = std::make_unique<GlobalQueueWorker>(this, workers_.size(),
                                                     device_flag);
      } else {
        worker = std::make_unique<DeviceQueueWorker>(this, workers_.size(),
                                                     device_flag);
      }

      if (!worker->Init(config.worker_config).ok()) {
        return absl::InternalError(absl::StrFormat(
            "Worker::Init() failed for worker : %s.", ToString(device_flag)));
      }

      BAND_LOG_INTERNAL(BAND_LOG_INFO, "%s worker is created.",
                        ToString(device_flag));
      worker->Start();
      workers_.push_back(std::move(worker));
      workers_waiting_[i] = 0;
      BAND_TRACER_ADD_WORKER(device_flag, workers_.back()->GetId());
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_WARNING, "%s worker is not created.",
                        ToString(device_flag));
    }
  }

  return absl::OkStatus();
}

Engine::Engine(ErrorReporter* error_reporeter) : IEngine(error_reporeter) {}

void Engine::UpdateWorkersWaiting() const {
  for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
    workers_waiting_[worker_id] = workers_[worker_id]->GetWaitingTime();
  }
}

WorkerWaitingTime Engine::GetWorkerWaitingTime() const {
  return workers_waiting_;
}

std::set<int> Engine::GetIdleWorkers() const {
  std::set<int> idle_workers;
  auto waiting_time = GetWorkerWaitingTime();
  for (auto worker_waiting : waiting_time) {
    if (worker_waiting.second == 0) {
      idle_workers.insert(worker_waiting.first);
    }
  }
  return idle_workers;
}

SubgraphKey Engine::GetLargestSubgraphKey(ModelId model_id,
                                          WorkerId worker_id) const {
  auto model_executor_it = model_executors_.find({model_id, worker_id});
  if (model_executor_it != model_executors_.end()) {
    return model_executor_it->second->GetLargestSubgraphKey();
  } else {
    return SubgraphKey();
  }
}

const ModelSpec* Engine::GetModelSpec(ModelId model_id) const {
  if (model_specs_.find(model_id) == model_specs_.end()) {
    return nullptr;
  } else {
    return &model_specs_.at(model_id);
  }
}

WorkerId Engine::GetModelWorker(ModelId model_id) const {
  return planner_->GetModelWorkerMap()[model_id];
}

bool Engine::IsBegin(const SubgraphKey& key) const {
  const ModelSpec* model_spec = GetModelSpec(key.GetModelId());
  if (!model_spec) {
    return false;
  }

  // if any of unit subgraph requires dependency, return false
  for (auto unit_index : key.GetUnitIndicesSet()) {
    if (model_spec->GetUnitSubgraphDependency(unit_index).any()) {
      return false;
    }
  }

  return true;
}

bool Engine::IsEnd(const SubgraphKey& key) const {
  const ModelSpec* model_spec = GetModelSpec(key.GetModelId());
  // check whether key has the last unit subgraph
  return model_spec &&
         (key.GetUnitIndices().test(model_spec->GetNumUnitSubgraphs() - 1) ||
          key.GetUnitIndices().none());
}

bool Engine::HasSubgraph(const SubgraphKey& key) const {
  auto model_executor_it =
      model_executors_.find({key.GetModelId(), key.GetWorkerId()});
  return model_executor_it != model_executors_.end() &&
         model_executor_it->second->HasSubgraph(key);
}

void Engine::ForEachSubgraph(
    std::function<void(const SubgraphKey&)> iterator) const {
  for (auto& model_executor : model_executors_) {
    model_executor.second->ForEachSubgraph(iterator);
  }
}

absl::Status Engine::Invoke(const SubgraphKey& key) {
  auto model_executor_it =
      model_executors_.find({key.GetModelId(), key.GetWorkerId()});
  if (model_executor_it == model_executors_.end()) {
    return absl::InternalError("Failed to find a subgraph key");
  }
  return model_executor_it->second->ExecuteSubgraph(key);
}

std::pair<SubgraphKey, int64_t> Engine::GetShortestLatency(
    ModelId model_id, BitMask resolved_unit_subgraphs, int64_t start_time,
    const std::map<WorkerId, int64_t>& worker_waiting) const {
  // lookup key for cache
  std::pair<ModelId, BitMask> cache_key = {model_id, resolved_unit_subgraphs};

  // check if it is safe to lookup the cache:
  // are all waiting times < start_time ?
  bool wait_time_is_stale = true;
  for (auto& pair : worker_waiting) {
    auto wait_time = pair.second;
    if (wait_time > start_time) {
      wait_time_is_stale = false;
    }
  }

  if (wait_time_is_stale) {
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
      auto& pair = it->second;
      // the stored latency value assumes a start_time of 0,
      // so we need to add our own start_time to the stored value to get the
      // correct return value
      return {pair.first, pair.second + start_time};
    }
  }

  std::vector<SubgraphKey> candidates =
      GetSubgraphCandidates(model_id, resolved_unit_subgraphs);

  auto bit_mask_comparator = [](const BitMask& lhs, const BitMask& rhs) {
    return lhs.to_ullong() < rhs.to_ullong();
  };

  std::map<BitMask, std::vector<SubgraphKey>, decltype(bit_mask_comparator)>
      unit_indicies_subgraphs(bit_mask_comparator);
  // group by unit indices
  for (const SubgraphKey& key : candidates) {
    unit_indicies_subgraphs[key.GetUnitIndices()].push_back(key);
  }

  std::pair<SubgraphKey, int64_t> subgraph_min_latency{
      {}, std::numeric_limits<int64_t>::max()};
  for (const auto& it : unit_indicies_subgraphs) {
    // first, filter out the subgraphs that take longer than others with the
    // same start/end indices, since there's no reason to pick them
    std::pair<SubgraphKey, int64_t> target_subgraph =
        GetShortestSubgraphKey(it.second, start_time, worker_waiting);

    std::pair<SubgraphKey, int64_t> local_min;
    if (IsEnd(target_subgraph.first)) {
      local_min = target_subgraph;
    } else {
      local_min = GetShortestLatency(
          model_id,
          resolved_unit_subgraphs | target_subgraph.first.GetUnitIndices(),
          target_subgraph.second, worker_waiting);
    }

    // check if this subgraph is better than the best one
    if (local_min.second < subgraph_min_latency.second) {
      // note the subgraph to return is the next immediate one (start_idx, XX),
      // but the latency to return is that of the final subgraph (XX, #ops)
      // hence, target_subgraph.first & local_min.second
      subgraph_min_latency.first = target_subgraph.first;
      subgraph_min_latency.second = local_min.second;
    }
  }

  if (wait_time_is_stale) {
    // if we've reached this point, then there shouldn't be an entry
    // for this key in the cache
    assert(cache_.find(cache_key) == cache_.end());
    // we are going to store the latency value for start_time == 0,
    // so do a sanity check for latency - start_time
    assert(subgraph_min_latency.second >= start_time);

    cache_[cache_key] = {subgraph_min_latency.first,
                         subgraph_min_latency.second - start_time};
  }

  return subgraph_min_latency;
}

std::pair<std::vector<SubgraphKey>, int64_t>
Engine::GetShortestLatencyWithUnitSubgraph(
    ModelId model_id, int start_unit_idx,
    const std::map<WorkerId, int64_t>& worker_waiting) const {
  const ModelSpec* model_spec = GetModelSpec(model_id);
  // vector for memoization during scheduling.
  // Each element is a pair of subgraph indices list and shortest latency.
  std::vector<std::pair<std::vector<SubgraphKey>, int64_t>> memo;
  const size_t num_unit_subgraphs = model_spec->GetNumUnitSubgraphs();
  memo.resize(num_unit_subgraphs);

  assert(start_unit_idx < num_unit_subgraphs);

  // Initialize memo.
  for (int i = 0; i < num_unit_subgraphs; ++i) {
    memo[i] = std::make_pair<std::vector<SubgraphKey>, int64_t>({}, INT_MAX);
  }

  // `i` and `j` refer to an unit subgraph idx.
  // A subgraph(i, j) consists of the unit subgraphs in [i, j].
  // The goal of the algorithm is to find the minimum expected latency;
  // `memo[k].second` is the minimum expected latency of the
  // subgraph(start_unit_idx, k). `memo[k].first` is the list of subgraph
  // indices of the best execution plan. So, the shortest expected latency of a
  // subgraph(start_unit_idx, num_unit_subgraphs - 1) is
  // `memo[num_unit_subgraphs - 1].second`.
  for (int j = start_unit_idx; j < num_unit_subgraphs; ++j) {
    std::pair<std::vector<SubgraphKey>, int64_t> local_min =
        std::make_pair<std::vector<SubgraphKey>, int64_t>({}, -1);
    for (int i = j; i >= start_unit_idx; --i) {
      // Check if the subgraph(i, j) is valid.
      if (unit_subgraphs_to_subgraph_keys_.find(model_id) ==
          unit_subgraphs_to_subgraph_keys_.end()) {
        continue;
      }
      if (unit_subgraphs_to_subgraph_keys_.at(model_id).find(i) ==
          unit_subgraphs_to_subgraph_keys_.at(model_id).end()) {
        continue;
      }
      if (unit_subgraphs_to_subgraph_keys_.at(model_id).at(i).find(j) ==
          unit_subgraphs_to_subgraph_keys_.at(model_id).at(i).end()) {
        continue;
      }

      // Search from the profile result of the unit subgraph.
      const auto& subgraph_keys =
          unit_subgraphs_to_subgraph_keys_.at(model_id).at(i).at(j);
      int64_t start = i > start_unit_idx ? memo[i - 1].second : 0;
      std::pair<SubgraphKey, int64_t> target_subgraph =
          GetShortestSubgraphKey(subgraph_keys, start, worker_waiting);

      if (local_min.second == -1 || target_subgraph.second < local_min.second) {
        if (i > start_unit_idx) {
          local_min.first = memo[i - 1].first;
          local_min.first.push_back(target_subgraph.first);
          local_min.second = target_subgraph.second;
        } else {
          local_min.first.clear();
          local_min.first.push_back(target_subgraph.first);
          local_min.second = target_subgraph.second;
        }
      }
    }
    memo[j] = local_min;
  }

  return memo[num_unit_subgraphs - 1];
}

std::pair<std::vector<SubgraphKey>, int64_t>
Engine::GetSubgraphWithShortestLatency(
    const Job& job, const std::map<WorkerId, int64_t>& worker_waiting) const {
  // TODO(dostos): figure out why we return a vector of keys?
  if (subgraph_config_.subgraph_preparation_type ==
      SubgraphPreparationType::kFallbackPerWorker) {
    auto pair = GetShortestLatency(job.model_id, job.resolved_unit_subgraphs, 0,
                                   worker_waiting);
    std::pair<std::vector<SubgraphKey>, int64_t> ret =
        std::pair<std::vector<SubgraphKey>, int64_t>({}, pair.second);
    ret.first.push_back(pair.first);
    return ret;
  } else {
    int start_unit_idx = 0;
    for (int i = 0; i < model_specs_.at(job.model_id).GetNumUnitSubgraphs();
         i++) {
      if (job.resolved_unit_subgraphs.test(i)) {
        start_unit_idx = i + 1;
      }
    }
    return GetShortestLatencyWithUnitSubgraph(job.model_id, start_unit_idx,
                                              worker_waiting);
  }
}

SubgraphKey Engine::GetSubgraphIdxSatisfyingSLO(
    const Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
    const std::set<WorkerId>& idle_workers) const {
  // TODO: implement this with SLO-based scheduler e.g., LSF
  BAND_NOT_IMPLEMENTED;
  return {};
}

std::vector<SubgraphKey> Engine::GetSubgraphCandidates(
    ModelId model_id, BitMask resolved_unit_subgraphs) const {
  std::vector<SubgraphKey> candidates;
  if (resolved_unit_subgraphs.none()) {
    for (const auto& model_executor : model_executors_) {
      if (model_executor.first.first == model_id) {
        model_executor.second->ForEachSubgraph(
            [this, &candidates](const SubgraphKey& key) {
              if (IsBegin(key)) {
                candidates.push_back(key);
              }
            });
      }
    }
  } else {
    for (const auto& model_executor : model_executors_) {
      if (model_executor.first.first == model_id) {
        model_executor.second->ForEachSubgraph([&](const SubgraphKey& key) {
          // skip if already executed
          if ((key.GetUnitIndices() & resolved_unit_subgraphs).any()) {
            return;
          }
          const BitMask external_dependencies =
              GetModelSpec(model_id)->GetUnitSubgraphDependency(
                  key.GetUnitIndices());
          // include if all external dependencies are resolved
          if (external_dependencies ==
              (external_dependencies & resolved_unit_subgraphs)) {
            candidates.push_back(key);
          }
        });
      }
    }
  }
  return candidates;
}

std::pair<SubgraphKey, int64_t> Engine::GetShortestSubgraphKey(
    const std::vector<SubgraphKey>& subgraph_keys, int64_t start_time,
    const std::map<WorkerId, int64_t>& worker_waiting) const {
  int64_t min_latency = std::numeric_limits<int64_t>::max();
  SubgraphKey min_key = {};

  for (const auto& key : subgraph_keys) {
    // TODO: safety check to avoid contention with profiler?
    int64_t waiting_time = worker_waiting.at(key.GetWorkerId());
    int64_t expected_latency = GetExpected(key);
    int64_t total = expected_latency + std::max(waiting_time, start_time);

    if (min_latency >= total) {
      min_latency = total;
      min_key = key;
    }
  }

  return {min_key, min_latency};
}

void Engine::Update(const SubgraphKey& key, int64_t new_value) {
  latency_estimator_->Update(key, new_value);
}

void Engine::UpdateWithEvent(const SubgraphKey& key, size_t event_id) {
  latency_estimator_->UpdateWithEvent(key, event_id);
  thermal_estimator_->UpdateWithEvent(key, event_id);
  frequency_latency_estimator_->UpdateWithEvent(key, event_id);
}

int64_t Engine::GetProfiled(const SubgraphKey& key) const {
  return latency_estimator_->GetProfiled(key);
}

int64_t Engine::GetExpected(const SubgraphKey& key) const {
  return latency_estimator_->GetExpected(key);
}

int64_t Engine::GetWorst(ModelId model_id) const {
  return latency_estimator_->GetWorst(model_id);
}

void Engine::Trigger() { planner_->Trigger(); }

int Engine::EnqueueRequest(Job job, bool push_front) {
  return planner_->EnqueueRequest(job, push_front);
}

std::vector<int> Engine::EnqueueBatch(std::vector<Job> jobs, bool push_front) {
  return planner_->EnqueueBatch(jobs, push_front);
}

void Engine::PrepareReenqueue(Job& job) { planner_->PrepareReenqueue(job); }

void Engine::EnqueueFinishedJob(Job& job) { planner_->EnqueueFinishedJob(job); }

bool Engine::EnqueueToWorker(const ScheduleAction& action) {
  return EnqueueToWorkerBatch(std::vector<ScheduleAction>{action});
}

bool Engine::EnqueueToWorkerBatch(
    const std::vector<ScheduleAction>& schedule_action) {
  return planner_->EnqueueToWorker(schedule_action);
}

const Worker* Engine::GetWorker(WorkerId id) const {
  if (id >= 0 && id < workers_.size()) {
    return workers_.at(id).get();
  } else {
    return nullptr;
  }
}

Worker* Engine::GetWorker(WorkerId id) {
  if (id >= 0 && id < workers_.size()) {
    return workers_[id].get();
  } else {
    return nullptr;
  }
}

absl::Status Engine::TryCopyInputTensors(const Job& job) {
  // Skip all tensor communication for compute only case.
  if (job.input_handle < 0) {
    return absl::OkStatus();
  }

  const SubgraphKey& key = job.subgraph_key;
  auto model_executor = GetModelExecutor(job.subgraph_key);
  std::set<int> unresolved_tensors(model_executor->GetInputs(key).begin(),
                                   model_executor->GetInputs(key).end());

  // Intermediate tensor communication
  for (auto subgraph_it = job.previous_subgraph_keys.cbegin();
       subgraph_it != job.previous_subgraph_keys.cend(); ++subgraph_it) {
    SubgraphKey preceded_subgraph_key = *subgraph_it;
    auto preceded_model_executor = GetModelExecutor(preceded_subgraph_key);

    for (int tensor_index :
         preceded_model_executor->GetOutputs(preceded_subgraph_key)) {
      if (unresolved_tensors.find(tensor_index) != unresolved_tensors.end()) {
        std::shared_ptr<interface::ITensorView> src =
            preceded_model_executor->GetTensorView(preceded_subgraph_key,
                                                   tensor_index);
        std::shared_ptr<interface::ITensorView> dst =
            model_executor->GetTensorView(key, tensor_index);

        if (!dst->CopyDataFrom(src.get()).ok()) {
          return absl::InternalError(
              absl::StrFormat("Tensor data copy failure from %s to %s",
                              src->GetName(), dst->GetName()));
        }

        unresolved_tensors.erase(tensor_index);
      }
    }
  }

  if (model_input_buffer_.find(job.model_id) == model_input_buffer_.end()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to find input tensor ring buffer for model %d", job.model_id));
  }

  auto input_buffer = model_input_buffer_[job.model_id].get();

  // Copy model input
  for (auto tensor_it = unresolved_tensors.begin();
       tensor_it != unresolved_tensors.end();) {
    int tensor_index = *tensor_it;
    if (input_buffer->IsTensorIndexValid(tensor_index)) {
      if (!input_buffer
               ->GetTensorFromHandle(
                   model_executor->GetTensorView(key, tensor_index).get(),
                   tensor_index, job.input_handle)
               .ok()) {
        return absl::InternalError(
            absl::StrFormat("Failed to copy input tensor %d for model %d",
                            tensor_index, job.model_id));
      }
      tensor_it = unresolved_tensors.erase(tensor_it);
    } else {
      ++tensor_it;
    }
  }

  if (!unresolved_tensors.empty()) {
    return absl::InternalError("Some tensors fail to be resolved.");
  }

  return absl::OkStatus();
}

absl::Status Engine::TryCopyOutputTensors(const Job& job) {
  // TODO: Subgraph execution

  // Compute only.
  if (job.output_handle < 0) {
    return absl::OkStatus();
  }

  const SubgraphKey& key = job.subgraph_key;
  auto model_executor = GetModelExecutor(job.subgraph_key);

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to find output tensor ring buffer for model %d", job.model_id));
  }

  auto output_buffer = model_output_buffer_[job.model_id].get();
  for (int tensor_index : model_executor->GetOutputs(key)) {
    if (output_buffer->IsTensorIndexValid(tensor_index)) {
      if (!output_buffer
               ->PutTensorToHandle(
                   model_executor->GetTensorView(key, tensor_index).get(),
                   tensor_index, job.output_handle)
               .ok()) {
        return absl::InternalError(
            absl::StrFormat("Failed to copy output tensor %d for model %d",
                            tensor_index, job.model_id));
      }
    }
  }

  return absl::OkStatus();
}

WorkerId Engine::GetDeviceWorkerId(DeviceFlag flag) const {
  for (WorkerId worker_id = 0; worker_id < workers_.size(); worker_id++) {
    if (workers_[worker_id]->GetDeviceFlag() == flag) {
      return worker_id;
    }
  }
  BAND_LOG_INTERNAL(BAND_LOG_WARNING, "Failed to find a worker for %s",
                    ToString(flag));
  return -1;
}

interface::IModelExecutor* Engine::GetModelExecutor(const SubgraphKey& key) {
  auto it = model_executors_.find({key.GetModelId(), key.GetWorkerId()});
  return it != model_executors_.end() ? it->second.get() : nullptr;
}

const interface::IModelExecutor* Engine::GetModelExecutor(
    const SubgraphKey& key) const {
  auto it = model_executors_.find({key.GetModelId(), key.GetWorkerId()});
  return it != model_executors_.end() ? it->second.get() : nullptr;
}

size_t Engine::BeginEvent() {
  size_t event_handle = -1;
  for (auto profiler : profilers_) {
    event_handle = profiler->BeginEvent();
  }
  return event_handle;
}

void Engine::EndEvent(size_t event_id) {
  for (int i = 0; i < profilers_.size(); i++) {
    profilers_[i]->EndEvent(event_id);
  }
}

absl::Status Engine::ProfileModel(ModelId model_id) {
  for (WorkerId worker_id = 0; worker_id < GetNumWorkers(); worker_id++) {
    Worker* worker = GetWorker(worker_id);
    worker->Pause();
    worker->Wait();
    std::thread profile_thread([&]() {
#if BAND_IS_MOBILE
      if (worker->GetWorkerThreadAffinity().NumEnabled() > 0) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to get worker thread affinity");
      }
      if (!SetCPUThreadAffinity(worker->GetWorkerThreadAffinity()).ok()) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to propagate thread affinity of worker id "
                          "%d to profile thread",
                          worker_id);
      }
#endif  // BAND_IS_MOBILE
      ForEachSubgraph([&](const SubgraphKey& subgraph_key) -> void {
        if (subgraph_key.GetWorkerId() != worker_id ||
            subgraph_key.GetModelId() != model_id) {
          return;
        }

        for (int i = 0; i < profile_config_.num_warmups; i++) {
          if (!Invoke(subgraph_key).ok()) {
            BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                              "Profiler failed to invoke largest subgraph of "
                              "model %d in worker %d during warmup.",
                              model_id, worker_id);
          }
        }

        for (int i = 0; i < profile_config_.num_runs; i++) {
          // All event handles must be the same.
          size_t event_id = BeginEvent();
          if (!Invoke(subgraph_key).ok()) {
            BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                              "Profiler failed to invoke largest subgraph of "
                              "model %d in worker %d during profiling.",
                              model_id, worker_id);
          }
          EndEvent(event_id);
          UpdateWithEvent(subgraph_key, event_id);
        }
      });
    });

    profile_thread.join();
    worker->Resume();
  }
  return absl::OkStatus();
}  // namespace band

}  // namespace band