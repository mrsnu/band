#include "band/c/common.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"

const char* BandSchedulerGetName(BandSchedulerType type) {
  switch (type) {
    case kBandFixedWorker:
      return "fixed_worker";
    case kBandRoundRobin:
      return "round_robin";
    case kBandShortestExpectedLatency:
      return "shortest_expected_latency";
    case kBandFixedWorkerGlobalQueue:
      return "fixed_worker_global_queue";
    case kBandHeterogeneousEarliestFinishTime:
      return "heterogeneous_earliest_finish_time";
    case kBandLeastSlackTimeFirst:
      return "least_slack_time_first";
    case kBandHeterogeneousEarliestFinishTimeReserved:
      return "heterogeneous_earliest_finish_time_reserved";
  }
  return "Unknown type";
}

BandSchedulerType BandSchedulerGetType(const char* name) {
  for (int i = 0; i < kBandNumSchedulerTypes; i++) {
    BandSchedulerType type = (BandSchedulerType)i;
    if (strncmp(BandSchedulerGetName(type), name,
                strlen(BandSchedulerGetName(type))) == 0) {
      return type;
    }
  }
  return kBandNumSchedulerTypes;
}

const char* BandSubgraphPreparationGetName(BandSubgraphPreparationType type) {
  switch (type) {
    case kBandNoFallbackSubgraph:
      return "no_fallback_subgraph";
    case kBandFallbackPerWorker:
      return "fallback_per_worker";
    case kBandUnitSubgraph:
      return "unit_subgraph";
    case kBandMergeUnitSubgraph:
      return "merge_unit_subgraph";
  }
  return "Unknown type";
}

BandSubgraphPreparationType BandSubgraphPreparationGetType(const char* name) {
  for (int i = 0; i < kBandNumSubgraphPreparationType; i++) {
    BandSubgraphPreparationType type = (BandSubgraphPreparationType)i;
    if (strncmp(BandSubgraphPreparationGetName(type), name,
                strlen(BandSubgraphPreparationGetName(type))) == 0) {
      return type;
    }
  }
  return kBandNumSubgraphPreparationType;
}

int BandIntArrayGetSizeInBytes(int size) {
  static BandIntArray dummy;
  return sizeof(dummy) + sizeof(dummy.data[0]) * size;
}

int BandIntArrayEqual(const BandIntArray* a, const BandIntArray* b) {
  if (a == b) return 1;
  if (a == NULL || b == NULL) return 0;
  return BandIntArrayEqualsArray(a, b->size, b->data);
}

int BandIntArrayEqualsArray(const BandIntArray* a, int b_size,
                            const int b_data[]) {
  if (a == NULL) return (b_size == 0);
  if (a->size != b_size) return 0;
  int i = 0;
  for (; i < a->size; i++)
    if (a->data[i] != b_data[i]) return 0;
  return 1;
}

BandIntArray* BandIntArrayCreate(int size) {
  BandIntArray* ret = (BandIntArray*)malloc(BandIntArrayGetSizeInBytes(size));
  ret->size = size;
  return ret;
}

BandIntArray* BandIntArrayCopy(const BandIntArray* src) {
  if (!src) return NULL;
  BandIntArray* ret = BandIntArrayCreate(src->size);
  if (ret) {
    memcpy(ret->data, src->data, src->size * sizeof(int));
  }
  return ret;
}

void BandIntArrayFree(BandIntArray* a) { free(a); }

int BandFloatArrayGetSizeInBytes(int size) {
  static BandFloatArray dummy;
  return sizeof(dummy) + sizeof(dummy.data[0]) * size;
}

BandFloatArray* BandFloatArrayCreate(int size) {
  BandFloatArray* ret =
      (BandFloatArray*)malloc(BandFloatArrayGetSizeInBytes(size));
  ret->size = size;
  return ret;
}

void BandFloatArrayFree(BandFloatArray* a) { free(a); }

void BandQuantizationFree(BandQuantization* quantization) {
  if (quantization->type == kBandAffineQuantization && quantization->params) {
    BandAffineQuantization* q_params =
        (BandAffineQuantization*)(quantization->params);
    if (q_params->scale) {
      BandFloatArrayFree(q_params->scale);
      q_params->scale = NULL;
    }
    if (q_params->zero_point) {
      BandIntArrayFree(q_params->zero_point);
      q_params->zero_point = NULL;
    }
    free(q_params);
  }
  quantization->params = NULL;
  quantization->type = kBandNoQuantization;
}

const char* BandTypeGetName(BandType type) {
  switch (type) {
    case kBandNoType:
      return "NOTYPE";
    case kBandFloat32:
      return "FLOAT32";
    case kBandInt16:
      return "INT16";
    case kBandInt32:
      return "INT32";
    case kBandUInt8:
      return "UINT8";
    case kBandInt8:
      return "INT8";
    case kBandInt64:
      return "INT64";
    case kBandBool:
      return "BOOL";
    case kBandComplex64:
      return "COMPLEX64";
    case kBandString:
      return "STRING";
    case kBandFloat16:
      return "FLOAT16";
    case kBandFloat64:
      return "FLOAT64";
  }
  return "Unknown type";
}

const char* BandDeviceGetName(BandDeviceFlags flag) {
  switch (flag) {
    case kBandCPU:
      return "CPU";
    case kBandGPU:
      return "GPU";
    case kBandDSP:
      return "DSP";
    case kBandNPU:
      return "NPU";
  }
  return "Unknown type";
}

BandDeviceFlags BandDeviceGetFlag(const char* name) {
  for (int i = 0; i < kBandNumDevices; i++) {
    BandDeviceFlags flag = (BandDeviceFlags)i;
    if (strncmp(BandDeviceGetName(flag), name,
                strlen(BandDeviceGetName(flag))) == 0) {
      return flag;
    }
  }
  return kBandNumDevices;
}

BandRequestOption BandGetDefaultRequestOption() { return {-1, true, -1, -1.f}; }

const char* BandBackendGetName(BandBackendType flag) {
  switch (flag) {
    case kBandTfLite:
      return "Tensorflow Lite";
  }
  return "Unknown type";
}

const BandBackendType BandBackendGetType(const char* name) {
  for (int i = 0; i < kBandNumBackendTypes; i++) {
    BandBackendType flag = (BandBackendType)i;
    if (strcmp(BandBackendGetName(flag), name) == 0) {
      return flag;
    }
  }
  return kBandNumBackendTypes;
}

const char* BandStatusGetName(BandStatus status) {
  switch (status) {
    case absl::OkStatus():
      return "Ok";
    case kBandDelegateError:
      return "DelegateError";
    case kBandError:
      return "Error";
  }
  return "Unknown type";
}