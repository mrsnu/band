#ifndef BAND_C_C_API_INTERNAL_H_
#define BAND_C_C_API_INTERNAL_H_

#include <list>
#include <memory>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/c/c_api_types.h"

struct BandConfigBuilder {
  Band::RuntimeConfigBuilder impl;
};

struct BandConfig {
  BandConfig(Band::RuntimeConfig config) : impl(config) {}
  Band::RuntimeConfig impl;
};

struct BandModel {
  BandModel() : impl(std::make_shared<Band::Model>()) {}
  std::shared_ptr<Band::Model> impl;
};

struct BandTensor {
  BandTensor(Band::Tensor* tensor)
      : impl(std::make_unique<Band::Tensor>(tensor)) {}
  std::unique_ptr<Band::Tensor> impl;
};

struct BandEngine {
  BandEngine(std::unique_ptr<Band::Engine> engine) : impl(std::move(engine)) {}
  // holds shared refs to registered models to guarantee the model's lifespan
  // matches with the engine.
  std::list<std::shared_ptr<Band::Model>> models;
  std::unique_ptr<Band::Engine> impl;
};

const char* BandBackendGetName(BandBackendType flag);
const BandBackendType BandBackendGetType(const char* name);

const char* BandStatusGetName(BandStatus status);

const char* BandSchedulerGetName(BandSchedulerType type);
BandSchedulerType BandSchedulerGetType(const char* name);

const char* BandSubgraphPreparationGetName(BandSubgraphPreparationType type);
BandSubgraphPreparationType BandSubgraphPreparationGetType(const char* name);

const char* BandTypeGetName(BandType type);

const char* BandQuantizationTypeGetName(BandQuantizationType type);

const char* BandDeviceGetName(BandDeviceFlags flag);
BandDeviceFlags BandDeviceGetFlag(const char* name);

#endif  // BAND_C_C_API_INTERNAL_H_
