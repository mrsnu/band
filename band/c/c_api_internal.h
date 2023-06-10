#ifndef BAND_C_C_API_INTERNAL_H_
#define BAND_C_C_API_INTERNAL_H_

#include <list>
#include <memory>

#include "band/buffer/buffer.h"
#include "band/c/c_api_type.h"
#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"


struct BandConfigBuilder {
  band::RuntimeConfigBuilder impl;
};

struct BandConfig {
  BandConfig(band::RuntimeConfig config) : impl(config) {}
  band::RuntimeConfig impl;
};

struct BandModel {
  BandModel() : impl(std::make_shared<band::Model>()) {}
  std::shared_ptr<band::Model> impl;
};

struct BandBuffer {
  BandBuffer();
  // lazily update the buffer data from c api functions
  std::shared_ptr<band::Buffer> impl;
};

struct BandTensor {
  BandTensor(band::Tensor* tensor)
      : impl(std::make_unique<band::Tensor>(tensor)) {}
  std::unique_ptr<band::Tensor> impl;
};

struct BandEngine {
  BandEngine(std::unique_ptr<band::Engine> engine) : impl(std::move(engine)) {}
  // holds shared refs to registered models to guarantee the model's lifespan
  // matches with the engine.
  std::list<std::shared_ptr<band::Model>> models;
  std::unique_ptr<band::Engine> impl;
};

const char* BandBackendToString(BandBackendType flag);
const BandBackendType BandBackendGetType(const char* name);

const char* BandStatusToString(BandStatus status);

const char* BandSchedulerToString(BandSchedulerType type);
BandSchedulerType BandSchedulerGetType(const char* name);

const char* BandSubgraphPreparationToString(BandSubgraphPreparationType type);
BandSubgraphPreparationType BandSubgraphPreparationGetType(const char* name);

const char* BandDataTypeToString(BandDataType type);

const char* BandQuantizationTypeToString(BandQuantizationType type);

const char* BandDeviceToString(BandDeviceFlag flag);
BandDeviceFlag BandDeviceGetFlag(const char* name);

#endif  // BAND_C_C_API_INTERNAL_H_
