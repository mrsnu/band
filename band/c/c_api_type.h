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
  kBandNumBackendType = 1
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

typedef enum BandCPUMaskFlag {
  kBandAll = 0,
  kBandLittle = 1,
  kBandBig = 2,
  kBandPrimary = 3,
  kBandNumCpuMask = 4
} BandCPUMaskFlag;

typedef enum BandSubgraphPreparationType {
  kBandNoFallbackSubgraph = 0,
  kBandFallbackPerWorker = 1,
  kBandUnitSubgraph = 2,
  kBandMergeUnitSubgraph = 3,
  kBandNumSubgraphPreparationType = 4,
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
  kBandNumDataType = 12,
} BandDataType;

typedef enum {
  // image format
  kBandGrayScale = 0,
  kBandRGB = 1,
  kBandRGBA = 2,
  kBandYV12 = 3,
  kBandYV21 = 4,
  kBandNV21 = 5,
  kBandNV12 = 6,
  // raw format, from tensor
  // internal format follows DataType
  kBandRaw = 7,
  kBandNumBufferFormat = 8,
} BandBufferFormat;

// Buffer content orientation follows EXIF specification. The name of
// each enum value defines the position of the 0th row and the 0th column of
// the image content. See http://jpegclub.org/exif_orientation.html for
// details.
typedef enum {
  kBandTopLeft = 1,
  kBandTopRight = 2,
  kBandBottomRight = 3,
  kBandBottomLeft = 4,
  kBandLeftTop = 5,
  kBandRightTop = 6,
  kBandRightBottom = 7,
  kBandLeftBottom = 8,
  kBandNumBufferOrientation = 9,
} BandBufferOrientation;

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
typedef enum BandDeviceFlag {
  kBandCPU = 0,
  kBandGPU = 1,
  kBandDSP = 2,
  kBandNPU = 3,
  kBandNumDeviceFlag = 4,
} BandDeviceFlag;

typedef enum BandConfigField {
  BAND_PROFILE_NUM_WARMUPS = 0,
  BAND_PROFILE_NUM_RUNS,
  BAND_PROFILE_LATENCY_SMOOTHING_FACTOR,
  BAND_PROFILE_FREQ_LATENCY_SMOOTHING_FACTOR,
  BAND_PROFILE_PATH,
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
} BandConfigField;

typedef enum BandImageProcessorBuilderField {
  BAND_CROP = 0,
  BAND_RESIZE = 1,
  BAND_ROTATE = 2,
  BAND_FLIP = 3,
  BAND_COLOR_SPACE_CONVERT = 4,
  BAND_NORMALIZE = 5,
  BAND_DATA_TYPE_CONVERT = 6,
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