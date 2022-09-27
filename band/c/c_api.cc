#include "band/c/c_api.h"

#include "band/c/c_api_type.h"
#include "band/interface/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

BandConfig* BandConfigCreate(const void* config_data, size_t config_size) {
  BandConfig* config = new BandConfig;
  Json::Value root;
  Json::Reader reader;
  if (reader.parse((const char*)config_data,
                   (const char*)config_data + config_size, root) &&
      Band::ParseRuntimeConfigFromJsonObject(root, config->impl) == kBandOk) {
    return config;
  } else {
    delete config;
    return nullptr;
  }
}

BandConfig* BandConfigCreateFromFile(const char* config_path) {
  BandConfig* config = new BandConfig;
  if (Band::ParseRuntimeConfigFromJson(config_path, config->impl) == kBandOk) {
    return config;
  } else {
    delete config;
    return nullptr;
  }
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
                                        int index) {
  auto input_indices =
      engine->impl->GetInputTensorIndices(model->impl->GetId());
  return new BandTensor(
      engine->impl->CreateTensor(model->impl->GetId(), input_indices[index]));
}

BandTensor* BandEngineCreateOutputTensor(BandEngine* engine, BandModel* model,
                                         int index) {
  auto output_indices =
      engine->impl->GetOutputTensorIndices(model->impl->GetId());
  return new BandTensor(
      engine->impl->CreateTensor(model->impl->GetId(), output_indices[index]));
}

std::vector<Band::Interface::ITensor*> BandTensorArrayToVec(
    BandTensor** tensors, int num_tensors) {
  std::vector<Band::Interface::ITensor*> vec(num_tensors);
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = tensors[i]->impl.get();
  }
  return vec;
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
  return {engine->impl->InvokeAsyncModel(
              model->impl->GetId(),
              BandTensorArrayToVec(
                  input_tensors, BandEngineGetNumInputTensors(engine, model))),
          model};
}

BandStatus BandEngineWait(BandEngine* engine, BandRequestHandle handle,
                          BandTensor** output_tensors) {
  return engine->impl->Wait(
      {handle.request_id},
      BandTensorArrayToVec(output_tensors, BandEngineGetNumOutputTensors(
                                               engine, handle.target_model)));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus