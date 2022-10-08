#include "band/engine.h"

#include "band/backend_factory.h"
#include "band/context.h"
#include "band/interface/interpreter.h"
#include "band/interface/tensor.h"
#include "band/interface/tensor_view.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/planner.h"
#include "band/profiler.h"
#include "band/tensor.h"
#include "band/worker.h"

namespace Band {
Engine::~Engine() {
  for (auto& worker : workers_) {
    worker.second->End();
  }

  for (auto& interpreter : interpreters_) {
    interpreter.second.reset();
  }

  for (auto& worker : workers_) {
    worker.second.reset();
  }

  planner_.reset();
}

absl::StatusOr<std::unique_ptr<Engine>> Engine::Create(
    const RuntimeConfig& config, ErrorReporter* error_reporter) {
  std::unique_ptr<Engine> engine_ptr(new Engine(error_reporter));
  if (!engine_ptr->Init(config).ok()) {
    return absl::InternalError("Failed to create engine.");
  }
  return std::move(engine_ptr);
}

absl::Status Engine::RegisterModel(Model* model) {
  const ModelId model_id = model->GetId();

  // Create internal interpreter per each supported backends
  for (BandBackendType backend_type : model->GetSupportedBackends()) {
    // Analyze model spec
    {
      auto interpreter = BackendFactory::CreateInterpreter(backend_type);
      if (!interpreter.ok()) {
        return interpreter.status();
      }
      std::unique_ptr<Interface::IInterpreter> interpreter_unique(
          interpreter.value());
      auto backend_model = model->GetBackendModel(backend_type);
      if (!backend_model.ok()) {
        return backend_model.status();
      }
      model_specs_[model_id] =
          interpreter_unique->InvestigateModelSpec(backend_model.value());
    }

    // Add whole-model subgraphs
    for (auto& id_worker : workers_) {
      auto interpreter = BackendFactory::CreateInterpreter(backend_type);
      if (!interpreter.ok()) {
        return interpreter.status();
      }
      std::unique_ptr<Interface::IInterpreter> interpreter_unique(
          interpreter.value());
      auto backend_model = model->GetBackendModel(backend_type);
      if (!backend_model.ok()) {
        return backend_model.status();
      }
      auto subgraph_key =
          interpreter_unique->FromModel(backend_model.value(), id_worker.first,
                                        id_worker.second->GetDeviceFlag());
      if (subgraph_key.ok()) {
        BAND_LOG_INTERNAL(BAND_LOG_INFO,
                          "Create interpreter for model %d worker %s", model_id,
                          BandDeviceGetName(id_worker.second->GetDeviceFlag()));
        subgraph_keys_.emplace(std::make_pair(id_worker.first, model_id),
                               subgraph_key.value());
        interpreters_.emplace(std::make_pair(id_worker.first, model_id),
                              std::move(interpreter_unique));
      }
    }

    bool need_fallback_subgraph =
        planner_->NeedFallbackSubgraphs() &&
        model_option_.subgraph_preparation_type_ != "no_fallback_subgraph";
    // TODO: Create a interpreter that represents <Interface::IModel,
    // DeviceFlag>

    // Initialize tensor ring buffer
    // Assumption: each backend model in Band::Model has the same input / output
    // tensor shapes
    {
      std::vector<std::shared_ptr<Interface::ITensor>> input_tensors;
      std::vector<std::shared_ptr<Interface::ITensor>> output_tensors;
      auto model_subgraph_key =
          GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
      if (!model_subgraph_key.ok()) {
        return model_subgraph_key.status();
      }
      auto primary_interpreter = GetInterpreter(model_subgraph_key.value());

      for (int input_tensor :
           primary_interpreter->GetInputs(model_subgraph_key.value())) {
        auto tensor_view = primary_interpreter->GetTensorView(
            model_subgraph_key.value(), input_tensor);
        input_tensors.push_back(tensor_view);
      }

      for (int output_tensor :
           primary_interpreter->GetOutputs(model_subgraph_key.value())) {
        auto tensor_view = primary_interpreter->GetTensorView(
            model_subgraph_key.value(), output_tensor);
        output_tensors.push_back(tensor_view);
      }

      model_input_buffer_.emplace(
          model->GetId(),
          std::make_unique<TensorRingBuffer>(
              error_reporter_, input_tensors,
              primary_interpreter->GetInputs(model_subgraph_key.value())));
      model_output_buffer_.emplace(
          model_id,
          std::make_unique<TensorRingBuffer>(
              error_reporter_, output_tensors,
              primary_interpreter->GetOutputs(model_subgraph_key.value())));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Tensor*> Engine::CreateTensor(ModelId model_id,
                                             int tensor_index) {
  // TODO: What if there are multiple backends?
  auto model_subgraph_key =
      GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  if (!model_subgraph_key.ok()) {
    return model_subgraph_key.status();
  }

  auto interpreter = GetInterpreter(model_subgraph_key.value());
  auto tensor_view =
      interpreter->GetTensorView(model_subgraph_key.value(), tensor_index);
  return new Tensor(tensor_view.get());
}

absl::StatusOr<std::vector<int>> Engine::GetOutputTensorIndices(
    ModelId model_id) const {
  auto model_subgraph_key =
      GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  if (!model_subgraph_key.ok()) {
    return model_subgraph_key.status();
  }
  auto interpreter = GetInterpreter(model_subgraph_key.value());
  return interpreter->GetOutputs(model_subgraph_key.value());
}

absl::StatusOr<std::vector<int>> Engine::GetInputTensorIndices(
    ModelId model_id) const {
  auto model_subgraph_key =
      GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  if (!model_subgraph_key.ok()) {
    return model_subgraph_key.status();
  }
  auto interpreter = GetInterpreter(model_subgraph_key.value());
  return interpreter->GetInputs(model_subgraph_key.value());
}

absl::Status Engine::InvokeSyncModel(ModelId model_id, Tensors inputs,
                                     Tensors outputs) {
  return Wait(InvokeAsyncModel(model_id, inputs), outputs);
}

absl::Status Engine::InvokeSyncModels(std::vector<ModelId> model_ids,
                                      std::vector<Tensors> inputs,
                                      std::vector<Tensors> outputs) {
  return Wait(InvokeAsyncModels(model_ids, inputs), outputs);
}

JobId Engine::InvokeAsyncModel(ModelId model_id, Tensors inputs) {
  std::vector<Tensors> input_tensors;
  if (inputs.size()) {
    input_tensors.push_back(inputs);
  }
  auto job_ids = InvokeAsyncModels({model_id}, input_tensors);
  return job_ids.size() == 1 ? job_ids[0] : -1;
}

std::vector<JobId> Engine::InvokeAsyncModels(std::vector<ModelId> model_ids,
                                             std::vector<Tensors> inputs) {
  std::vector<Job> jobs;

  for (size_t i = 0; i < model_ids.size(); i++) {
    Job job(model_ids[i]);
    if (i < inputs.size()) {
      int input_handle = model_input_buffer_[model_ids[i]]->Alloc();
      if (!model_input_buffer_[model_ids[i]]
               ->PutTensorsToHandle(inputs[i], input_handle)
               .ok()) {
        BAND_REPORT_ERROR(error_reporter_, "Input copy failure for model %d",
                          model_ids[i]);
        return {};
      }
      job.input_handle = input_handle;
      job.output_handle = model_output_buffer_[model_ids[i]]->Alloc();
    }
    jobs.push_back(job);
  }

  for (ModelId model_id : model_ids) {
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
  planner_->Wait(job_ids);
  for (size_t i = 0; i < outputs.size(); i++) {
    BAND_ENSURE_STATUS(GetOutputTensors(job_ids[i], outputs[i]));
  }
  return absl::OkStatus();
}

absl::Status Engine::GetOutputTensors(JobId job_id, Tensors outputs) {
  Job job = planner_->GetFinishedJob(job_id);

  if (outputs.empty() || job_id == -1) {
    BAND_REPORT_ERROR(error_reporter_,
                      "Invalid job id / num outputs to copy: (%d, %d)", job_id,
                      outputs.size());
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid job id / num outputs to copy: (%d, %d)",
                        job_id, outputs.size()));
  }

  // Not finished or invalidated
  if (job.job_id == -1) {
    return absl::InternalError("Job is not finished or invalidated.");
  }

  if (job.output_handle == -1) {
    BAND_REPORT_ERROR(error_reporter_, "Invalid output handle : %d",
                      job.output_handle);
    return absl::InternalError(
        absl::StrFormat("Invalid output handle : %d", job.output_handle));
  }

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    BAND_REPORT_ERROR(error_reporter_, "Invalid model id : %d", job.model_id);
    return absl::InternalError(
        absl::StrFormat("Invalid model id : %d", job.model_id));
  }

  BAND_ENSURE_STATUS(model_output_buffer_.at(job.model_id)
                         ->GetTensorsFromHandle(outputs, job.output_handle));
  return absl::OkStatus();
}

void Engine::SetEndInvokeFunction(
    std::function<void(int, absl::Status)> on_end_invoke) {
  planner_->SetEndInvokeFunction(on_end_invoke);
}

absl::Status Engine::Init(const RuntimeConfig& config) {
  planner_ = std::make_unique<Planner>(this);

  BAND_ENSURE_STATUS(planner_->Init(config.planner_config));

  // TODO: Update interpreter config to something
  {
    model_option_.minimum_subgraph_size_ = config.minimum_subgraph_size;
    model_option_.subgraph_preparation_type_ = config.subgraph_preparation_type;

    if (planner_->NeedProfile()) {
      profiler_ = std::make_unique<Profiler>();
      BAND_ENSURE_STATUS(profiler_->Init(config.profile_config));
    }

    const BandCPUMaskFlags cpu_mask =
        static_cast<BandCPUMaskFlags>(config.cpu_mask);
    auto cpu_mask_set = BandCPUMaskGetSet(cpu_mask);

    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Set affinity to %s cores.",
                      BandCPUMaskGetName(cpu_mask));

    BAND_ENSURE_STATUS(SetCPUThreadAffinity(cpu_mask_set));
  }

  // Search for all available backends, devices
  std::set<BandDeviceFlags> valid_devices;
  auto valid_backends = BackendFactory::GetAvailableBackends();
  for (auto backend : valid_backends) {
    auto backend_util = BackendFactory::GetBackendUtil(backend);
    if (!backend_util.ok()) {
      return backend_util.status();
    }
    auto backend_devices = backend_util.value()->GetAvailableDevices();
    valid_devices.insert(backend_devices.begin(), backend_devices.end());
  }

  auto& potential_workers = config.worker_config.workers;
  for (int i = 0; i < potential_workers.size(); i++) {
    BandDeviceFlags device_flag = potential_workers[i];
    if (valid_devices.find(device_flag) != valid_devices.end()) {
      std::unique_ptr<Worker> worker;
      if (planner_->GetWorkerType() == kBandGlobalQueue) {
        worker = std::make_unique<GlobalQueueWorker>(this, device_flag);
      } else {
        worker = std::make_unique<DeviceQueueWorker>(this, device_flag);
      }

      if (!worker->Init(config.worker_config, workers_.size()).ok()) {
        error_reporter_->Report("Worker::Init() failed for worker : %s.",
                                BandDeviceGetName(device_flag));
        exit(-1);
      }

      BAND_LOG_INTERNAL(BAND_LOG_INFO, "%s worker is created.",
                        BandDeviceGetName(device_flag));
      worker->Start();
      workers_[i] = std::move(worker);
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_WARNING, "%s worker is not created.",
                        BandDeviceGetName(device_flag));
    }
  }

  return absl::OkStatus();
}

Engine::Engine(ErrorReporter* error_reporeter) : Context(error_reporeter) {}

void Engine::UpdateWorkerWaitingTime() const {
  for (auto worker_it = workers_.begin(); worker_it != workers_.end();
       worker_it++) {
    workers_waiting_[worker_it->first] = worker_it->second->GetWaitingTime();
  }
}

const WorkerWaitingTime& Engine::GetWorkerWaitingTime() const {
  return workers_waiting_;
}

std::set<int> Engine::GetIdleWorkers() const {
  std::set<int> idle_workers;

  for (auto worker_it = workers_.begin(); worker_it != workers_.end();
       worker_it++) {
    if (!worker_it->second->HasJob()) {
      idle_workers.insert(worker_it->first);
    }
  }
  return idle_workers;
}

absl::StatusOr<SubgraphKey> Engine::GetModelSubgraphKey(
    ModelId model_id, WorkerId worker_id) const {
  return subgraph_keys_.at({worker_id, model_id});
}

bool Engine::IsEnd(const SubgraphKey& key) const {
  // TODO: subgraph support
  return true;
}

absl::Status Engine::Invoke(const SubgraphKey& key) {
  auto interpreter_it =
      interpreters_.find({key.GetWorkerId(), key.GetModelId()});
  if (interpreter_it != interpreters_.end()) {
    return interpreter_it->second->InvokeSubgraph(key);
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_WARNING, "Failed to find a subgraph key");
    return absl::NotFoundError("Failed to find a subgraph key");
  }
}

// TODO(widiba03304): replace `int` into `SubgraphKey` when ready.
std::pair<int, int64_t> Engine::GetShortestLatency(
    int model_id, std::set<int> resolved_tensors, int64_t start_time,
    const std::map<WorkerId, int64_t>& worker_waiting,
    SubgraphKey preceded_subgraph_index) const {
  BAND_NOT_IMPLEMENTED;
  return {};
}

// TODO(widiba03304): replace `int` into `SubgraphKey` when ready.
std::pair<std::vector<int>, int64_t> Engine::GetShortestLatencyWithUnitSubgraph(
    int model_id, int start_unit_idx,
    const std::map<WorkerId, int64_t>& worker_waiting) const {
  std::pair<std::vector<int>, int64_t> result;
  BAND_NOT_IMPLEMENTED;
  return result;
}

// TODO(widiba03304): replace `int` into `SubgraphKey` when ready.
std::pair<std::vector<int>, int64_t> Engine::GetSubgraphWithShortestLatency(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting) const {
  std::pair<std::vector<int>, int64_t> result;
  BAND_NOT_IMPLEMENTED;
  return result;
}

// TODO(widiba03304): replace `int` into `SubgraphKey` when ready.
int Engine::GetSubgraphIdxSatisfyingSLO(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
    const std::set<WorkerId>& idle_workers) const {
  BAND_NOT_IMPLEMENTED;
  return {};
}

void Engine::UpdateLatency(const SubgraphKey& key, int64_t latency) {
  if (profiler_) profiler_->UpdateLatency(key, latency);
}

int64_t Engine::GetProfiled(const SubgraphKey& key) const {
  return profiler_ ? profiler_->GetProfiled(key) : 0;
}

int64_t Engine::GetExpected(const SubgraphKey& key) const {
  return profiler_ ? profiler_->GetExpected(key) : 0;
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

absl::StatusOr<Worker*> Engine::GetWorker(WorkerId id) {
  auto it = workers_.find(id);
  if (it == workers_.end()) {
    absl::InvalidArgumentError("Invalid worker id.");
  }
  return it->second.get();
}

absl::Status Engine::TryCopyInputTensors(const Job& job) {
  // Skip all tensor communication for compute only case.
  if (job.input_handle < 0) {
    return absl::OkStatus();
  }

  // TODO: Subgraph execution
  const SubgraphKey& key = *job.subgraph_key;
  auto interpeter = GetInterpreter(key);
  auto unresolved_inputs = interpeter->GetInputs(key);
  std::set<int> unresolved_tensors(unresolved_inputs.begin(),
                                   unresolved_inputs.end());

  // // Intermediate tensor communication
  // for (auto subgraph_it = job.previous_subgraph_indices.cbegin();
  //      subgraph_it != job.previous_subgraph_indices.cend();
  //      ++subgraph_it) {
  //   int preceded_subgraph_index = *subgraph_it;
  //   Subgraph *preceded_subgraph =
  //       interpreter->subgraph(preceded_subgraph_index);

  //   for (int tensor_index : preceded_subgraph->outputs()) {
  //     if (unresolved_tensors.find(tensor_index) !=
  //     unresolved_tensors.end())
  //     {
  //       const ITensorView *src = preceded_subgraph->tensor(tensor_index);
  //       ITensorView *dst = subgraph->tensor(tensor_index);

  //       if (ITensorDataCopy(src, dst) == kBandError) {
  //         BAND_REPORT_ERROR(GetErrorReporter(),
  //                           "Tensor data copy failure from %s to %s",
  //                           src->name, dst->name);
  //         return kBandError;
  //       }

  //       unresolved_tensors.erase(tensor_index);
  //     }
  //   }
  // }

  if (model_input_buffer_.find(job.model_id) == model_input_buffer_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Failed to find input tensor ring buffer for model %d",
                      job.model_id);
    return absl::NotFoundError(absl::StrFormat(
        "Failed to find input tensor ring buffer for model %d", job.model_id));
  }

  auto input_buffer = model_input_buffer_[job.model_id].get();

  // Copy model input
  for (auto tensor_it = unresolved_tensors.begin();
       tensor_it != unresolved_tensors.end();) {
    int tensor_index = *tensor_it;
    if (input_buffer->IsTensorIndexValid(tensor_index)) {
      auto tensor_view = interpeter->GetTensorView(key, tensor_index);
      if (!input_buffer
               ->GetTensorFromHandle(tensor_view.get(), tensor_index,
                                     job.input_handle)
               .ok()) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to copy input tensor %d for model %d",
                          tensor_index, job.model_id);
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
    return absl::InternalError("Unresolved tensors remain.");
  }
  return absl::OkStatus();
}

absl::Status Engine::TryCopyOutputTensors(const Job& job) {
  // TODO: Subgraph execution

  // Compute only.
  if (job.output_handle < 0) {
    return absl::OkStatus();
  }

  const SubgraphKey& key = *job.subgraph_key;
  auto interpeter = GetInterpreter(key);

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Failed to find output tensor ring buffer for model %d",
                      job.model_id);
    return absl::NotFoundError(absl::StrFormat(
        "Failed to find output tensor ring buffer for model %d", job.model_id));
  }

  auto output_buffer = model_output_buffer_[job.model_id].get();
  auto outputs = interpeter->GetOutputs(key);
  for (int tensor_index : outputs) {
    if (output_buffer->IsTensorIndexValid(tensor_index)) {
      if (!output_buffer
               ->PutTensorToHandle(
                   interpeter->GetTensorView(key, tensor_index).get(),
                   tensor_index, job.output_handle)
               .ok()) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to copy output tensor %d for model %d",
                          tensor_index, job.model_id);
        return absl::InternalError(
            absl::StrFormat("Failed to copy output tensor %d for model %d",
                            tensor_index, job.model_id));
      }
    }
  }

  return absl::OkStatus();
}

WorkerId Engine::GetDeviceWorkerId(BandDeviceFlags flag) const {
  for (auto& it : workers_) {
    if (it.second->GetDeviceFlag() == flag) {
      return it.first;
    }
  }
  BAND_LOG_INTERNAL(BAND_LOG_WARNING, "Failed to find a worker for %s",
                    BandDeviceGetName(flag));
  return -1;
}

Interface::IInterpreter* Engine::GetInterpreter(const SubgraphKey& key) {
  auto it = interpreters_.find({key.GetWorkerId(), key.GetModelId()});
  return it->second.get();
}

const Interface::IInterpreter* Engine::GetInterpreter(
    const SubgraphKey& key) const {
  auto it = interpreters_.find({key.GetWorkerId(), key.GetModelId()});
  return it->second.get();
}
}  // namespace Band