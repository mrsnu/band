#ifndef BAND_C_C_API_TYPE_H_
#define BAND_C_C_API_TYPE_H_

#include <list>
#include <memory>

#include "band/config_builder.h"
#include "band/config.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"

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

struct BandRequestHandle {
  int request_id;
  BandModel* target_model;
};

typedef enum BandConfigField {
  BAND_PROFILE_ONLINE = 0,
  BAND_PROFILE_NUM_WARMUPS = 1,
  BAND_PROFILE_NUM_RUNS = 2,
  BAND_PROFILE_COPY_COMPUTATION_RATIO = 3,
  BAND_PROFILE_SMOOTHING_FACTOR = 4,
  BAND_PROFILE_DATA_PATH = 5,
  BAND_PLANNER_SCHEDULE_WINDOW_SIZE = 6,
  BAND_PLANNER_SCHEDULERS = 7,
  BAND_PLANNER_CPU_MASK = 8,
  BAND_PLANNER_LOG_PATH = 9,
  BAND_WORKER_ADDITIONAL_WORKERS = 10,
  BAND_WORKER_CPU_MASKS = 11,
  BAND_WORKER_NUM_THREADS = 12,
  BAND_WORKER_ALLOW_WORKSTEAL = 13,
  BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS = 14,
  BAND_MINIMUM_SUBGRAPH_SIZE =15,
  BAND_SUBGRAPH_PREPARATION_TYPE = 16,
  BAND_CPU_MASK = 17,
} BandConfigField;

#endif