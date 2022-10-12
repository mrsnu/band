#include "band/c/c_api.h"

#include "band/c/c_api_internal.h"
#include "band/interface/tensor.h"
#include "band/interface/tensor_view.h"

std::vector<Band::Interface::ITensor*> BandTensorArrayToVec(
    BandTensor** tensors, int num_tensors) {
  std::vector<Band::Interface::ITensor*> vec(num_tensors);
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = tensors[i]->impl.get();
  }
  return vec;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

BandConfigBuilder* BandConfigBuilderCreate() {
  return new BandConfigBuilder; 
}

void BandAddConfig(BandConfigBuilder* b, int field, int count, ...) {
  // TODO(widiba03304): Error handling should be properly done.
  va_list vl;
  va_start(vl, count);
  BandConfigField new_field = static_cast<BandConfigField>(field);
  switch (field) {
    case BAND_PROFILE_ONLINE: {
      bool arg = va_arg(vl, int);
      b->impl.AddOnline(arg);
    } break;
    case BAND_PROFILE_NUM_WARMUPS: {
      int arg = va_arg(vl, int);
      b->impl.AddNumWarmups(arg);
    } break;
    case BAND_PROFILE_NUM_RUNS: {
      int arg = va_arg(vl, int);
      b->impl.AddNumRuns(arg);
    } break;
    case BAND_PROFILE_COPY_COMPUTATION_RATIO: {
      std::vector<int> copy_computation_ratio(count);
      for (int i = 0; i < count; i++) {
        copy_computation_ratio[i] = va_arg(vl, int);
      }
      b->impl.AddCopyComputationRatio(copy_computation_ratio);
    } break;
    case BAND_PROFILE_SMOOTHING_FACTOR: {
      float arg = va_arg(vl, double);
      b->impl.AddSmoothingFactor(arg);
    } break;
    case BAND_PROFILE_DATA_PATH: {
      char* arg = va_arg(vl, char*);
      b->impl.AddProfileDataPath(arg);
    } break;
    case BAND_PLANNER_SCHEDULE_WINDOW_SIZE: {
      int arg = va_arg(vl, int);
      b->impl.AddScheduleWindowSize(arg);
    } break;
    case BAND_PLANNER_SCHEDULERS: {
      std::vector<BandSchedulerType> schedulers(count);
      for (int i = 0; i < count; i++) {
        schedulers[i] = static_cast<BandSchedulerType>(va_arg(vl, int));
      }
      b->impl.AddSchedulers(schedulers);
    } break;
    case BAND_PLANNER_CPU_MASK: {
      int arg = va_arg(vl, int);
      b->impl.AddPlannerCPUMask(static_cast<BandCPUMaskFlags>(arg));
    } break;
    case BAND_PLANNER_LOG_PATH: {
      char* arg = va_arg(vl, char*);
      b->impl.AddPlannerLogPath(arg);
    } break;
    case BAND_WORKER_WORKERS: {
      std::vector<BandDeviceFlags> workers(count);
      for (int i = 0; i < count; i++) {
        int temp = va_arg(vl, int);
        workers[i] = static_cast<BandDeviceFlags>(temp);
      }
      b->impl.AddWorkers(workers);
    } break;
    case BAND_WORKER_CPU_MASKS: {
      std::vector<BandCPUMaskFlags> cpu_masks(count);
      for (int i = 0; i < count; i++) {
        cpu_masks[i] = static_cast<BandCPUMaskFlags>(va_arg(vl, int));
      }
      b->impl.AddWorkerCPUMasks(cpu_masks);
    } break;
    case BAND_WORKER_NUM_THREADS: {
      std::vector<int> num_threads(count);
      for (int i = 0; i < count; i++) {
        num_threads[i] = va_arg(vl, int);
      }
      b->impl.AddWorkerNumThreads(num_threads);
    } break;
    case BAND_WORKER_ALLOW_WORKSTEAL: {
      bool arg = va_arg(vl, int);
      b->impl.AddAllowWorkSteal(arg);
    } break;
    case BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS: {
      int arg = va_arg(vl, int);
      b->impl.AddAvailabilityCheckIntervalMs(arg);
    } break;
    case BAND_MINIMUM_SUBGRAPH_SIZE: {
      int arg = va_arg(vl, int);
      b->impl.AddMinimumSubgraphSize(arg);
    } break;
    case BAND_SUBGRAPH_PREPARATION_TYPE: {
      int arg = va_arg(vl, int);
      b->impl.AddSubgraphPreparationType(
          static_cast<BandSubgraphPreparationType>(arg));
    } break;
    case BAND_CPU_MASK: {
      int arg = va_arg(vl, int);
      b->impl.AddCPUMask(static_cast<BandCPUMaskFlags>(arg));
    } break;
  }
  va_end(vl);
}

void BandConfigBuilderDelete(BandConfigBuilder* b) { delete b; }

BandConfig* BandConfigCreate(BandConfigBuilder* b) {
  // TODO(widiba03304): Error handling is not properly done here.
  BandConfig* config = new BandConfig(b->impl.Build());
  return config;
}

void BandConfigDelete(BandConfig* config) { delete config; }

BandModel* BandModelCreate() { return new BandModel; }

void BandModelDelete(BandModel* model) { delete model; }

BandStatus BandModelAddFromBuffer(BandModel* model,
                                  BandBackendType backend_type,
                                  const void* model_data, size_t model_size) {
  return model->impl->FromBuffer(backend_type, (const char*)model_data,
                                 model_size);
}

BandStatus BandModelAddFromFile(BandModel* model, BandBackendType backend_type,
                                const char* model_path) {
  return model->impl->FromPath(backend_type, model_path);
}

void BandTensorDelete(BandTensor* tensor) { delete tensor; }

BandType BandTensorGetType(BandTensor* tensor) {
  return tensor->impl->GetType();
}

void* BandTensorGetData(BandTensor* tensor) { return tensor->impl->GetData(); }

int* BandTensorGetDims(BandTensor* tensor) {
  return tensor->impl->GetDims().data();
}

size_t BandTensorGetBytes(BandTensor* tensor) {
  return tensor->impl->GetBytes();
}

const char* BandTensorGetName(BandTensor* tensor) {
  return tensor->impl->GetName();
}

BandQuantization BandTensorGetQuantization(BandTensor* tensor) {
  return tensor->impl->GetQuantization();
}

BandEngine* BandEngineCreate(BandConfig* config) {
  return new BandEngine(std::move(Band::Engine::Create(config->impl)));
}
void BandEngineDelete(BandEngine* engine) { delete engine; }

BandStatus BandEngineRegisterModel(BandEngine* engine, BandModel* model) {
  auto status = engine->impl->RegisterModel(model->impl.get());
  if (status == kBandOk) {
    engine->models.push_back(model->impl);
  }
  return status;
}

int BandEngineGetNumInputTensors(BandEngine* engine, BandModel* model) {
  return engine->impl->GetInputTensorIndices(model->impl->GetId()).size();
}

int BandEngineGetNumOutputTensors(BandEngine* engine, BandModel* model) {
  return engine->impl->GetOutputTensorIndices(model->impl->GetId()).size();
}

BandTensor* BandEngineCreateInputTensor(BandEngine* engine, BandModel* model,
                                        size_t index) {
  auto input_indices =
      engine->impl->GetInputTensorIndices(model->impl->GetId());
  return new BandTensor(
      engine->impl->CreateTensor(model->impl->GetId(), input_indices[index]));
}

BandTensor* BandEngineCreateOutputTensor(BandEngine* engine, BandModel* model,
                                         size_t index) {
  auto output_indices =
      engine->impl->GetOutputTensorIndices(model->impl->GetId());
  return new BandTensor(
      engine->impl->CreateTensor(model->impl->GetId(), output_indices[index]));
}

BandStatus BandEngineRequestSync(BandEngine* engine, BandModel* model,
                                 BandTensor** input_tensors,
                                 BandTensor** output_tensors) {
  return engine->impl->InvokeSyncModel(
      model->impl->GetId(),
      BandTensorArrayToVec(input_tensors,
                           BandEngineGetNumInputTensors(engine, model)),
      BandTensorArrayToVec(output_tensors,
                           BandEngineGetNumOutputTensors(engine, model)));
}

BandRequestHandle BandEngineRequestAsync(BandEngine* engine, BandModel* model,
                                         BandTensor** input_tensors) {
  return engine->impl->InvokeAsyncModel(
      model->impl->GetId(),
      BandTensorArrayToVec(input_tensors,
                           BandEngineGetNumInputTensors(engine, model)));
}

BandStatus BandEngineWait(BandEngine* engine, BandRequestHandle handle,
                          BandTensor** output_tensors, size_t num_outputs) {
  return engine->impl->Wait(handle,
                            BandTensorArrayToVec(output_tensors, num_outputs));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
