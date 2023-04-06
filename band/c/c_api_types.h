#ifndef BAND_C_C_API_TYPE_H_
#define BAND_C_C_API_TYPE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum BandBackendType {
  kBandTfLite = 0,
  kBandNumBackendTypes = 1
} BandBackendType;

typedef enum BandStatus {
  kBandOk = 0,
  kBandError = 1,
  kBandDelegateError = 2
} BandStatus;

typedef enum BandWorkerType {
  kBandDeviceQueue = 1 << 0,
  kBandGlobalQueue = 1 << 1,
} BandWorkerType;

typedef enum BandSchedulerType {
  kBandFixedWorker = 0,
  kBandRoundRobin = 1,
  kBandShortestExpectedLatency = 2,
  kBandFixedWorkerGlobalQueue = 3,
  kBandHeterogeneousEarliestFinishTime = 4,
  kBandLeastSlackTimeFirst = 5,
  kBandHeterogeneousEarliestFinishTimeReserved = 6,
  kBandNumSchedulerType = 7
} BandSchedulerType;

typedef enum BandCPUMaskFlags {
  kBandAll = 0,
  kBandLittle = 1,
  kBandBig = 2,
  kBandPrimary = 3,
  kNumCpuMasks = 4
} BandCPUMaskFlags;

typedef enum BandSubgraphPreparationType {
  kBandNoFallbackSubgraph = 0,
  kBandFallbackPerWorker = 1,
  kBandUnitSubgraph = 2,
  kBandMergeUnitSubgraph = 3,
  kNumSubgraphPreparationType = 4,
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
  kBandFloat32 = 1,
  kBandInt32 = 2,
  kBandUInt8 = 3,
  kBandInt64 = 4,
  kBandString = 5,
  kBandBool = 6,
  kBandInt16 = 7,
  kBandComplex64 = 8,
  kBandInt8 = 9,
  kBandFloat16 = 10,
  kBandFloat64 = 11,
} BandType;

// Supported Quantization Types.
typedef enum BandQuantizationType {
  // No quantization.
  kBandNoQuantization = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to BandAffineQuantization.
  kBandAffineQuantization = 1,
} BandQuantizationType;

// TODO #23, #30
// Add additional devices for HTA, NPU
typedef enum BandDeviceFlags {
  kBandCPU = 0,
  kBandGPU = 1,
  kBandDSP = 2,
  kBandNPU = 3,
  kNumDevices = 4,
} BandDeviceFlags;

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
  BAND_WORKER_WORKERS = 10,
  BAND_WORKER_CPU_MASKS = 11,
  BAND_WORKER_NUM_THREADS = 12,
  BAND_WORKER_ALLOW_WORKSTEAL = 13,
  BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS = 14,
  BAND_MINIMUM_SUBGRAPH_SIZE = 15,
  BAND_SUBGRAPH_PREPARATION_TYPE = 16,
  BAND_CPU_MASK = 17,
} BandConfigField;

typedef struct BandRequestOption {
  int target_worker;
  bool require_callback;
  int slo_us;
  float slo_scale;
} BandRequestOption;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // BAND_C_C_API_TYPE_H_