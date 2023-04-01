#include "band/c/c_api_internal.h"

#include <stdlib.h>
#include <string.h>

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
    default: {}
  }
  return "Unknown type";
}

BandSchedulerType BandSchedulerGetType(const char* name) {
  for (int i = 0; i < kNumSchedulerTypes; i++) {
    BandSchedulerType type = (BandSchedulerType)i;
    if (strncmp(BandSchedulerGetName(type), name,
                strlen(BandSchedulerGetName(type))) == 0) {
      return type;
    }
  }
  return kNumSchedulerTypes;
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
    default: {}
  }
  return "Unknown type";
}

BandSubgraphPreparationType BandSubgraphPreparationGetType(const char* name) {
  for (int i = 0; i < kNumSubgraphPreparationType; i++) {
    BandSubgraphPreparationType type = (BandSubgraphPreparationType)i;
    if (strncmp(BandSubgraphPreparationGetName(type), name,
                strlen(BandSubgraphPreparationGetName(type))) == 0) {
      return type;
    }
  }
  return kNumSubgraphPreparationType;
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
    default: {}
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
    default: {}
  }
  return "Unknown type";
}

BandDeviceFlags BandDeviceGetFlag(const char* name) {
  for (int i = 0; i < kNumDevices; i++) {
    BandDeviceFlags flag = (BandDeviceFlags)i;
    if (strncmp(BandDeviceGetName(flag), name,
                strlen(BandDeviceGetName(flag))) == 0) {
      return flag;
    }
  }
  return kNumDevices;
}

const char* BandBackendGetName(BandBackendType flag) {
  switch (flag) {
    case kBandTfLite:
      return "Tensorflow Lite";
    default: {}
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
    case kBandOk:
      return "Ok";
    case kBandDelegateError:
      return "DelegateError";
    case kBandError:
      return "Error";
    default: {}
  }
  return "Unknown type";
}