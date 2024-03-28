/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_C_C_API_INTERNAL_H_
#define BAND_C_C_API_INTERNAL_H_

#include <list>
#include <memory>

#include "band/buffer/buffer.h"
#include "band/buffer/image_processor.h"
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
  BandBuffer() : impl(nullptr) {}
  // lazily update the buffer data from c api functions
  std::shared_ptr<band::Buffer> impl;
};

struct BandImageProcessorBuilder {
  BandImageProcessorBuilder()
      : impl(std::make_unique<band::ImageProcessorBuilder>()) {}
  std::unique_ptr<band::ImageProcessorBuilder> impl;
};

struct BandImageProcessor {
  BandImageProcessor(std::unique_ptr<band::BufferProcessor> processor)
      : impl(std::move(processor)) {}
  std::unique_ptr<band::BufferProcessor> impl;
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
