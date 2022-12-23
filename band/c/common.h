#ifndef BAND_C_COMMON_H
#define BAND_C_COMMON_H

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

const char* BandBackendGetName(BandBackendType flag);
const BandBackendType BandBackendGetType(const char* name);

typedef enum BandStatus {
  kBandOk = 0,
  kBandError = 1,
  kBandDelegateError = 2
} BandStatus;

const char* BandStatusGetName(BandStatus status);

typedef enum {
  kBandAll = 0,
  kBandLittle = 1,
  kBandBig = 2,
  kBandPrimary = 3,
  kBandNumCpuMasks = 4
} BandCPUMaskFlags;

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
  kNumSchedulerTypes = 7
} BandSchedulerType;

typedef enum BandSubgraphPreparationType {
  kBandNoFallbackSubgraph = 0,
  kBandFallbackPerDevice = 1,
  kBandUnitSubgraph = 2,
  kBandMergeUnitSubgraph = 3,
  kBandNumSubgraphPreparationType = 4,
} BandSubgraphPreparationType;

const char* BandSubgraphPreparationGetName(BandSubgraphPreparationType type);

// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct BandIntArray {
  int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
#if (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
     __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON)
  int data[0];
#else
  int data[];
#endif
} BandIntArray;

// Given the size (number of elements) in a BandIntArray, calculate its size
// in bytes.
int BandIntArrayGetSizeInBytes(int size);

// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using BandIntArrayFree().
BandIntArray* BandIntArrayCreate(int size);

// Check if two intarrays are equal. Returns 1 if they are equal, 0 otherwise.
int BandIntArrayEqual(const BandIntArray* a, const BandIntArray* b);

// Check if an intarray equals an array. Returns 1 if equals, 0 otherwise.
int BandIntArrayEqualsArray(const BandIntArray* a, int b_size,
                            const int b_data[]);

// Create a copy of an array passed as `src`.
// You are expected to free memory with BandIntArrayFree
BandIntArray* BandIntArrayCopy(const BandIntArray* src);

// Free memory of array `a`.
void BandIntArrayFree(BandIntArray* a);

// Fixed size list of floats. Used for per-channel quantization.
typedef struct BandFloatArray {
  int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
// This also applies to the toolchain used for Qualcomm Hexagon DSPs.
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
    __GNUC_MINOR__ >= 1
  float data[0];
#else
  float data[];
#endif
} BandFloatArray;

// Given the size (number of elements) in a BandFloatArray, calculate its size
// in bytes.
int BandFloatArrayGetSizeInBytes(int size);

// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using BandFloatArrayFree().
BandFloatArray* BandFloatArrayCreate(int size);

// Free memory of array `a`.
void BandFloatArrayFree(BandFloatArray* a);

// Since we must not depend on any libraries, define a minimal subset of
// error macros while avoiding names that have pre-conceived meanings like
// assert and check.

// Try to make all reporting calls through BAND_KERNEL_LOG rather than
// calling the context->ReportError function directly, so that message strings
// can be stripped out if the binary size needs to be severely optimized.
#ifndef BAND_STRIP_ERROR_STRINGS
#define BAND_KERNEL_LOG(context, ...)               \
  do {                                              \
    (context)->ReportError((context), __VA_ARGS__); \
  } while (false)

#define BAND_MAYBE_KERNEL_LOG(context, ...)           \
  do {                                                \
    if ((context) != nullptr) {                       \
      (context)->ReportError((context), __VA_ARGS__); \
    }                                                 \
  } while (false)
#else  // BAND_STRIP_ERROR_STRINGS
#define BAND_KERNEL_LOG(context, ...)
#define BAND_MAYBE_KERNEL_LOG(context, ...)
#endif  // BAND_STRIP_ERROR_STRINGS

#define BAND_ENSURE_FORMATTED_MSG(context, value, ...) \
  do {                                                 \
    if (!(value)) {                                    \
      BAND_KERNEL_LOG((context), __VA_ARGS__);         \
      return kBandError;                               \
    }                                                  \
  } while (0)

// Check whether value is true, and if not return kBandError from
// the current function (and report the error string msg).
#define BAND_ENSURE_MSG(context, value, msg)        \
  do {                                              \
    if (!(value)) {                                 \
      BAND_KERNEL_LOG((context), __FILE__ " " msg); \
      return kBandError;                            \
    }                                               \
  } while (0)

// Check whether the value `a` is true, and if not return kBandError from
// the current function, while also reporting the location of the error.
#define BAND_ENSURE(context, a)                                                \
  do {                                                                         \
    if (!(a)) {                                                                \
      BAND_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, __LINE__, \
                      #a);                                                     \
      return kBandError;                                                       \
    }                                                                          \
  } while (0)

#define BAND_ENSURE_STATUS(a) \
  do {                        \
    const BandStatus s = (a); \
    if (s != kBandOk) {       \
      return s;               \
    }                         \
  } while (0)

// Check whether the value `a == b` is true, and if not return kBandError from
// the current function, while also reporting the location of the error.
// `a` and `b` may be evaluated more than once, so no side effects or
// extremely expensive computations should be done.
// NOTE: Use BAND_ENSURE_TYPES_EQ if comparing BandTypes.
#define BAND_ENSURE_EQ(context, a, b)                                   \
  do {                                                                  \
    if ((a) != (b)) {                                                   \
      BAND_KERNEL_LOG((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                      __LINE__, #a, #b, (a), (b));                      \
      return kBandError;                                                \
    }                                                                   \
  } while (0)

#define BAND_ENSURE_TYPES_EQ(context, a, b)                             \
  do {                                                                  \
    if ((a) != (b)) {                                                   \
      BAND_KERNEL_LOG((context), "%s:%d %s != %s (%s != %s)", __FILE__, \
                      __LINE__, #a, #b, BandTypeGetName(a),             \
                      BandTypeGetName(b));                              \
      return kBandError;                                                \
    }                                                                   \
  } while (0)

#define BAND_ENSURE_OK(context, status) \
  do {                                  \
    const BandStatus s = (status);      \
    if ((s) != kBandOk) {               \
      return s;                         \
    }                                   \
  } while (0)

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

// Return the name of a given type, for error reporting purposes.
const char* BandTypeGetName(BandType type);

// SupportedQuantizationTypes.
typedef enum BandQuantizationType {
  // No quantization.
  kBandNoQuantization = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to BandAffineQuantization.
  kBandAffineQuantization = 1,
} BandQuantizationType;

// Structure specifying the quantization used by the tensor, if-any.
typedef struct BandQuantization {
  // The type of quantization held by params.
  BandQuantizationType type;
  // Holds a reference to one of the quantization param structures specified
  // below.
  void* params;
} BandQuantization;

void BandQuantizationFree(BandQuantization* quantization);

// Legacy. Will be deprecated in favor of BandAffineQuantization.
// If per-layer quantization is specified this field will still be populated in
// addition to BandAffineQuantization.
// Parameters for asymmetric quantization. Quantized values can be converted
// back to float using:
//     real_value = scale * (quantized_value - zero_point)
typedef struct BandQuantizationParams {
  float scale;
  int32_t zero_point;
} BandQuantizationParams;

// Parameters for asymmetric quantization across a dimension (i.e per output
// channel quantization).
// quantized_dimension specifies which dimension the scales and zero_points
// correspond to.
// For a particular value in quantized_dimension, quantized values can be
// converted back to float using:
//     real_value = scale * (quantized_value - zero_point)
typedef struct BandAffineQuantization {
  BandFloatArray* scale;
  BandIntArray* zero_point;
  int32_t quantized_dimension;
} BandAffineQuantization;

// Optional parameters for model request
// `target_worker`: designate the target worker for a request.
// [default : -1 (not specified)] This option requires the FixedWorkerScheduler.
// `require_callback`: report if OnEndRequest is specified in an engine
// [default: true]
// `slo_us` and `slo_scale`: specifying an SLO value for a model.
// Setting `slo_scale` will make the SLO =  slo_scale * profiled latency of
// that model. `slo_scale` will be ignored if `slo_us` is given
// (i.e., no reason to specify both options). [default : -1 (not specified)]
typedef struct BandRequestOption {
  int target_worker;
  bool require_callback;
  int slo_us;
  float slo_scale;
} BandRequestOption;

BandRequestOption BandGetDefaultRequestOption();

// TODO #23, #30
// Add additional devices for HTA, NPU
typedef enum {
  kBandCPU = 0,
  kBandGPU = 1,
  kBandDSP = 2,
  kBandNPU = 3,
  kBandNumDevices = 4,
} BandDeviceFlags;

const char* BandDeviceGetName(BandDeviceFlags flag);
BandDeviceFlags BandDeviceGetFlag(const char* name);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // BAND_C_COMMON_H
