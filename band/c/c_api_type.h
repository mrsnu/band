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

#ifndef BAND_C_C_TYPE_H_
#define BAND_C_C_TYPE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum BandBackendType {
  kBandTfLite = 0,
  kBandNumBackendType
} BandBackendType;

typedef enum BandStatus {
  kBandOk = 0,
  kBandError,
  kBandDelegateError
} BandStatus;

typedef enum BandWorkerType {
  kBandDeviceQueue = 1 << 0,
  kBandGlobalQueue = 1 << 1,
} BandWorkerType;

typedef enum BandSchedulerType {
  kBandFixedWorker = 0,
  kBandRoundRobin,
  kBandShortestExpectedLatency,
  kBandFixedWorkerGlobalQueue,
  kBandHeterogeneousEarliestFinishTime,
  kBandLeastSlackTimeFirst,
  kBandHeterogeneousEarliestFinishTimeReserved,
  kBandNumSchedulerType
} BandSchedulerType;

typedef enum BandCPUMaskFlag {
  kBandAll = 0,
  kBandLittle,
  kBandBig,
  kBandPrimary,
  kBandNumCpuMask
} BandCPUMaskFlag;

typedef enum BandSubgraphPreparationType {
  kBandNoFallbackSubgraph = 0,
  kBandFallbackPerWorker,
  kBandUnitSubgraph,
  kBandMergeUnitSubgraph,
  kBandNumSubgraphPreparationType
} BandSubgraphPreparationType;

// Single-precision complex data type compatible with the C99 definition.
typedef struct BandComplex64 {
  float re, im;  // real and imaginary parts, respectively.
} BandComplex64;

// Half precision data type compatible with the C99 definition.
typedef struct BandFloat16 {
  uint16_t data;
} BandFloat16;

// Types supported by tensor
typedef enum {
  kBandNoType = 0,
  kBandFloat32,
  kBandInt32,
  kBandUInt8,
  kBandInt64,
  kBandString,
  kBandBool,
  kBandInt16,
  kBandComplex64,
  kBandInt8,
  kBandFloat16,
  kBandFloat64,
  kBandNumDataType,
} BandDataType;

typedef enum {
  // image format
  kBandGrayScale = 0,
  kBandRGB,
  kBandRGBA,
  kBandYV12,
  kBandYV21,
  kBandNV21,
  kBandNV12,
  // raw format, from tensor
  // internal format follows DataType
  kBandRaw,
  kBandNumBufferFormat,
} BandBufferFormat;

// Buffer content orientation follows EXIF specification. The name of
// each enum value defines the position of the 0th row and the 0th column of
// the image content. See http://jpegclub.org/exif_orientation.html for
// details.
typedef enum {
  kBandTopLeft = 1,
  kBandTopRight,
  kBandBottomRight,
  kBandBottomLeft,
  kBandLeftTop,
  kBandRightTop,
  kBandRightBottom,
  kBandLeftBottom,
  kBandNumBufferOrientation,
} BandBufferOrientation;

// Supported Quantization Types.
typedef enum BandQuantizationType {
  // No quantization.
  kBandNoQuantization = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to BandAffineQuantization.
  kBandAffineQuantization1,
} BandQuantizationType;

// TODO #23, #30
// Add additional devices for HTA, NPU
typedef enum BandDeviceFlag {
  kBandCPU = 0,
  kBandGPU,
  kBandDSP,
  kBandNPU,
  kBandNumDeviceFlag,
} BandDeviceFlag;

typedef enum BandConfigField {
  BAND_PROFILE_ONLINE = 0,
  BAND_PROFILE_NUM_WARMUPS,
  BAND_PROFILE_NUM_RUNS,
  BAND_PROFILE_SMOOTHING_FACTOR,
  BAND_PROFILE_DATA_PATH,
  BAND_PLANNER_SCHEDULE_WINDOW_SIZE,
  BAND_PLANNER_SCHEDULERS,
  BAND_PLANNER_CPU_MASK,
  BAND_PLANNER_LOG_PATH,
  BAND_WORKER_WORKERS,
  BAND_WORKER_CPU_MASKS,
  BAND_WORKER_NUM_THREADS,
  BAND_WORKER_ALLOW_WORKSTEAL,
  BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS,
  BAND_MINIMUM_SUBGRAPH_SIZE,
  BAND_SUBGRAPH_PREPARATION_TYPE,
  BAND_CPU_MASK,
  BAND_RESOURCE_MONITOR_DEVICE_PATH,
  BAND_RESOURCE_MONITOR_INTERVAL_MS,
  BAND_RESOURCE_MONITOR_LOG_PATH,
} BandConfigField;

typedef enum BandImageProcessorBuilderField {
  BAND_CROP = 0,
  BAND_RESIZE,
  BAND_ROTATE,
  BAND_FLIP,
  BAND_COLOR_SPACE_CONVERT,
  BAND_NORMALIZE,
  BAND_DATA_TYPE_CONVERT
} BandImageProcessorBuilderField;

typedef struct BandRequestOption {
  int target_worker;
  bool require_callback;
  int slo_us;
  float slo_scale;
} BandRequestOption;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // BAND_C_C_TYPE_H_