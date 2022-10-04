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

std::unique_ptr<Engine> Engine::Create(const RuntimeConfig& config,
                                       ErrorReporter* error_reporter) {
  std::unique_ptr<Engine> engine_ptr(new Engine(error_reporter));
  return engine_ptr->Init(config) == kBandOk ? std::move(engine_ptr) : nullptr;
}

BandStatus Engine::RegisterModel(Model* model) {
  const ModelId model_id = model->GetId();

  // Create internal interpreter per each supported backends
  for (BandBackendType backend_type : model->GetSupportedBackends()) {
    // Analyze model spec
    {
      std::unique_ptr<Interface::IInterpreter> interpreter(
          BackendFactory::CreateInterpreter(backend_type));
      model_specs_[model_id] = interpreter->InvestigateModelSpec(
          model->GetBackendModel(backend_type));
    }

    // Add whole-model subgraphs
    for (auto& id_worker : workers_) {
      std::unique_ptr<Interface::IInterpreter> interpreter(
          BackendFactory::CreateInterpreter(backend_type));
      if (interpreter->FromModel(
              model->GetBackendModel(backend_type), id_worker.first,
              id_worker.second->GetDeviceFlag()) == kBandOk) {
        BAND_LOG_INTERNAL(BAND_LOG_INFO,
                          "Create interpreter for model %d worker %s", model_id,
                          BandDeviceGetName(id_worker.second->GetDeviceFlag()));
        interpreters_[{id_worker.first, model_id}] = std::move(interpreter);
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
      Interface::IInterpreter* primary_interpreter =
          GetInterpreter(model_subgraph_key);

      for (int input_tensor :
           primary_interpreter->GetInputs(model_subgraph_key)) {
        input_tensors.push_back(primary_interpreter->GetTensorView(
            model_subgraph_key, input_tensor));
      }

      for (int output_tensor :
           primary_interpreter->GetOutputs(model_subgraph_key)) {
        output_tensors.push_back(primary_interpreter->GetTensorView(
            model_subgraph_key, output_tensor));
      }

      model_input_buffer_.emplace(
          model->GetId(),
          std::make_unique<TensorRingBuffer>(
              error_reporter_, input_tensors,
              primary_interpreter->GetInputs(model_subgraph_key)));
      model_output_buffer_.emplace(
          model_id, std::make_unique<TensorRingBuffer>(
                        error_reporter_, output_tensors,
                        primary_interpreter->GetOutputs(model_subgraph_key)));
    }
  }

  return kBandOk;
}

Tensor* Engine::CreateTensor(ModelId model_id, int tensor_index) {
  // TODO: What if there are multiple backends?
  SubgraphKey model_subgraph_key =
      GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));

  if (Interface::IInterpreter* interpreter =
          GetInterpreter(model_subgraph_key)) {
    return new Tensor(
        interpreter->GetTensorView(model_subgraph_key, tensor_index).get());
  } else {
    return nullptr;
  }
}

std::vector<int> Engine::GetOutputTensorIndices(ModelId model_id) const {
  SubgraphKey model_subgraph_key =
      GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  const Interface::IInterpreter* interpreter =
      GetInterpreter(model_subgraph_key);
  return interpreter ? interpreter->GetOutputs(model_subgraph_key)
                     : std::vector<int>();
}

std::vector<int> Engine::GetInputTensorIndices(ModelId model_id) const {
  SubgraphKey model_subgraph_key =
      GetModelSubgraphKey(model_id, GetDeviceWorkerId(kBandCPU));
  const Interface::IInterpreter* interpreter =
      GetInterpreter(model_subgraph_key);
  return interpreter ? interpreter->GetInputs(model_subgraph_key)
                     : std::vector<int>();
}

BandStatus Engine::InvokeSyncModel(ModelId model_id, Tensors inputs,
                                   Tensors outputs) {
  return Wait(InvokeAsyncModel(model_id, inputs), outputs);
}

BandStatus Engine::InvokeSyncModels(std::vector<ModelId> model_ids,
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
      if (model_input_buffer_[model_ids[i]]->PutTensorsToHandle(
              inputs[i], input_handle) != kBandOk) {
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

BandStatus Engine::Wait(JobId job_id, Tensors outputs) {
  std::vector<Tensors> output_tensors;
  if (outputs.size()) {
    output_tensors.push_back(outputs);
  }
  return Wait(std::vector<JobId>({job_id}), output_tensors);
}

BandStatus Engine::Wait(std::vector<JobId> job_ids,
                        std::vector<Tensors> outputs) {
  planner_->Wait(job_ids);
  for (size_t i = 0; i < outputs.size(); i++) {
    BAND_ENSURE_STATUS(GetOutputTensors(job_ids[i], outputs[i]));
  }
  return kBandOk;
}

BandStatus Engine::GetOutputTensors(JobId job_id, Tensors outputs) {
  Job job = planner_->GetFinishedJob(job_id);

  if (outputs.empty() || job_id == -1) {
    BAND_REPORT_ERROR(error_reporter_,
                      "Invalid job id / num outputs to copy: (%d, %d)", job_id,
                      outputs.size());
    return kBandError;
  }

  // Not finished or invalidated
  if (job.job_id == -1) {
    return kBandError;
  }

  if (job.output_handle == -1) {
    BAND_REPORT_ERROR(error_reporter_, "Invalid output handle : %d",
                      job.output_handle);
    return kBandError;
  }

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    BAND_REPORT_ERROR(error_reporter_, "Invalid model id : %d", job.model_id);
    return kBandError;
  }

  BAND_ENSURE_STATUS(model_output_buffer_.at(job.model_id)
                         ->GetTensorsFromHandle(outputs, job.output_handle));
  return kBandOk;
}

void Engine::SetEndInvokeFunction(
    std::function<void(int, BandStatus)> on_end_invoke) {
  planner_->SetEndInvokeFunction(on_end_invoke);
}

BandStatus Engine::Init(const RuntimeConfig& config) {
  planner_ = std::make_unique<Planner>(this);

  BAND_ENSURE_STATUS(planner_->Init(config.planner_config));

  // TODO: Update interpreter config to something
  {
    model_option_.minimum_subgraph_size_ =
        config.interpreter_config.minimum_subgraph_size;
    model_option_.subgraph_preparation_type_ =
        config.interpreter_config.subgraph_preparation_type;

    if (planner_->NeedProfile()) {
      profiler_ = std::make_unique<Profiler>();
      BAND_ENSURE_STATUS(
          profiler_->Init(config.interpreter_config.profile_config));
    }

    const BandCPUMaskFlags cpu_mask =
        static_cast<BandCPUMaskFlags>(config.interpreter_config.cpu_masks);
    auto cpu_mask_set = BandCPUMaskGetSet(cpu_mask);

    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Set affinity to %s cores.",
                      BandCPUMaskGetName(cpu_mask));

    BAND_ENSURE_STATUS(SetCPUThreadAffinity(cpu_mask_set));
  }

  // Search for all available backends, devices
  std::set<BandDeviceFlags> valid_devices;
  auto valid_backends = BackendFactory::GetAvailableBackends();
  for (auto backend : valid_backends) {
    auto backend_devices =
        BackendFactory::GetBackendUtil(backend)->GetAvailableDevices();
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
      WorkerId worker_id = workers_.size();
      if (worker->Init(config.worker_config, worker_id) != kBandOk) {
        error_reporter_->Report("Worker::Init() failed for worker : %s.",
                                BandDeviceGetName(device_flag));
        exit(-1);
      }

      BAND_LOG_INTERNAL(BAND_LOG_INFO, "%s worker is created.",
                        BandDeviceGetName(device_flag));
      worker->Start();
      workers_[worker_id] = std::move(worker);
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_WARNING, "%s worker is not created.",
                        BandDeviceGetName(device_flag));
    }
  }

  // Instantiate and register a model for each model_config
  std::vector<int> assigned_workers(workers_.size());
  for(int i=0; i<assigned_workers.size(); i++) assigned_workers[i] = 0;
  for(auto model_config : config.interpreter_config.models_config){
    std::shared_ptr<Model> model_ptr = std::make_shared<Model>();
    ModelId model_id;
    // Create a model for each valid backend
    // TODO(juimdpp): selectively support a single backend (based on config)
    for(auto backend: valid_backends){
      if(model_ptr->FromPath(backend, model_config.model_fname.c_str()) != kBandOk){
        error_reporter_->Report("Model %s could not be instantiated for %s.",
        model_config.model_fname, BandBackendGetName(backend));
      }
    }
    
    // Save model and its config
    model_id = model_ptr->GetId();
    model_configs_[model_id] = model_config;
    models_.emplace(model_id, model_ptr);

    // For each unassigned worker whose device matches the model's requested device, assign it to the model
    // In case # of models > # of workers, Planner::TryUpdateModelWorkerMapping will reassign later
    int i=0;
    for(auto worker_it = workers_.begin(); worker_it != workers_.end(); worker_it++, i++){
      if(assigned_workers[i] == 0 && worker_it->second->GetDeviceFlag() == model_config.device){
        planner_->GetModelWorkerMap()[model_id] = i;
        assigned_workers[i] = 1;
      }
    }   

    // Register model
    if(RegisterModel(model_ptr.get()) != kBandOk){
      error_reporter_->Report("Model %s could not be registered.", model_config.model_fname);
    }
  }
  return kBandOk;
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

SubgraphKey Engine::GetModelSubgraphKey(ModelId model_id,
                                        WorkerId worker_id) const {
  auto interpreter_it = interpreters_.find({worker_id, model_id});
  if (interpreter_it != interpreters_.end()) {
    return interpreter_it->second->GetModelSubgraphKey(model_id);
  } else {
    // TODO: report error
    return SubgraphKey();
  }
}

WorkerId Engine::GetModelWorker(ModelId model_id) const {
   return planner_->GetModelWorkerMap()[model_id];
}

bool Engine::IsEnd(const SubgraphKey& key) const {
  // TODO: subgraph support
  return true;
}

BandStatus Engine::Invoke(const SubgraphKey& key) {
  auto interpreter_it =
      interpreters_.find({key.GetWorkerId(), key.GetModelId()});
  if (interpreter_it != interpreters_.end()) {
    return interpreter_it->second->InvokeSubgraph(key);
  } else {
    BAND_LOG_INTERNAL(BAND_LOG_WARNING, "Failed to find a subgraph key");
    return kBandError;
  }
}

std::pair<SubgraphKey, int64_t> Engine::GetShortestLatency(
    int model_id, std::set<int> resolved_tensors, int64_t start_time,
    const std::map<WorkerId, int64_t>& worker_waiting,
    SubgraphKey preceded_subgraph_index) const {
  BAND_NOT_IMPLEMENTED;
  return {};
}

std::pair<std::vector<SubgraphKey>, int64_t>
Engine::GetShortestLatencyWithUnitSubgraph(
    int model_id, int start_unit_idx,
    const std::map<WorkerId, int64_t>& worker_waiting) const {
  std::pair<std::vector<SubgraphKey>, int64_t> result;
  BAND_NOT_IMPLEMENTED;
  return result;
}

std::pair<std::vector<SubgraphKey>, int64_t>
Engine::GetSubgraphWithShortestLatency(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting) const {
  std::pair<std::vector<SubgraphKey>, int64_t> result;
  BAND_NOT_IMPLEMENTED;
  return result;
}

SubgraphKey Engine::GetSubgraphIdxSatisfyingSLO(
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

Worker* Engine::GetWorker(WorkerId id) {
  auto it = workers_.find(id);
  return it != workers_.end() ? it->second.get() : nullptr;
}

BandStatus Engine::TryCopyInputTensors(const Job& job) {
  // Skip all tensor communication for compute only case.
  if (job.input_handle < 0) {
    return kBandOk;
  }

  // TODO: Subgraph execution
  const SubgraphKey& key = job.subgraph_key;
  auto interpeter = GetInterpreter(job.subgraph_key);
  std::set<int> unresolved_tensors(interpeter->GetInputs(key).begin(),
                                   interpeter->GetInputs(key).end());

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
    return kBandError;
  }

  auto input_buffer = model_input_buffer_[job.model_id].get();

  // Copy model input
  for (auto tensor_it = unresolved_tensors.begin();
       tensor_it != unresolved_tensors.end();) {
    int tensor_index = *tensor_it;
    if (input_buffer->IsTensorIndexValid(tensor_index)) {
      if (input_buffer->GetTensorFromHandle(
              interpeter->GetTensorView(key, tensor_index).get(), tensor_index,
              job.input_handle) != kBandOk) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to copy input tensor %d for model %d",
                          tensor_index, job.model_id);
        return kBandError;
      }
      tensor_it = unresolved_tensors.erase(tensor_it);
    } else {
      ++tensor_it;
    }
  }

  if (unresolved_tensors.empty()) {
    return kBandOk;
  } else {
    return kBandError;
  }

  return kBandOk;
}

BandStatus Engine::TryCopyOutputTensors(const Job& job) {
  // TODO: Subgraph execution

  // Compute only.
  if (job.output_handle < 0) {
    return kBandOk;
  }

  const SubgraphKey& key = job.subgraph_key;
  auto interpeter = GetInterpreter(job.subgraph_key);

  if (model_output_buffer_.find(job.model_id) == model_output_buffer_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                      "Failed to find output tensor ring buffer for model %d",
                      job.model_id);
    return kBandError;
  }

  auto output_buffer = model_output_buffer_[job.model_id].get();
  for (int tensor_index : interpeter->GetOutputs(key)) {
    if (output_buffer->IsTensorIndexValid(tensor_index)) {
      if (output_buffer->PutTensorToHandle(
              interpeter->GetTensorView(key, tensor_index).get(), tensor_index,
              job.output_handle) != kBandOk) {
        BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                          "Failed to copy output tensor %d for model %d",
                          tensor_index, job.model_id);
        return kBandError;
      }
    }
  }

  return kBandOk;
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
  return it != interpreters_.end() ? it->second.get() : nullptr;
}

const Interface::IInterpreter* Engine::GetInterpreter(
    const SubgraphKey& key) const {
  auto it = interpreters_.find({key.GetWorkerId(), key.GetModelId()});
  return it != interpreters_.end() ? it->second.get() : nullptr;
}
}  // namespace Band