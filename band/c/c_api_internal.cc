#include "band/c/c_api_internal.h"

#include <stdlib.h>
#include <string.h>

const char* BandSchedulerToString(BandSchedulerType type) {
  switch (type) {
    case kBandFixedWorker:
      return "fixed_worker";
    case kBandRoundRobin:
      return "round_robin";
    case kBandFixedWorkerGlobalQueue:
      return "fixed_worker_global_queue";
    case kBandHeterogeneousEarliestFinishTime:
      return "heterogeneous_earliest_finish_time";
    case kBandLeastSlackTimeFirst:
      return "least_slack_time_first";
    case kBandHeterogeneousEarliestFinishTimeReserved:
      return "heterogeneous_earliest_finish_time_reserved";
    default: {
    }
  }
  return "Unknown type";
}

BandSchedulerType BandSchedulerGetType(const char* name) {
  for (int i = 0; i < kBandNumSchedulerType; i++) {
    BandSchedulerType type = (BandSchedulerType)i;
    if (strncmp(BandSchedulerToString(type), name,
                strlen(BandSchedulerToString(type))) == 0) {
      return type;
    }
  }
  return kBandNumSchedulerType;
}

const char* BandSubgraphPreparationToString(BandSubgraphPreparationType type) {
  switch (type) {
    case kBandNoFallbackSubgraph:
      return "no_fallback_subgraph";
    case kBandUnitSubgraph:
      return "unit_subgraph";
    case kBandMergeUnitSubgraph:
      return "merge_unit_subgraph";
    default: {
    }
  }
  return "Unknown type";
}

BandSubgraphPreparationType BandSubgraphPreparationGetType(const char* name) {
  for (int i = 0; i < kBandNumSubgraphPreparationType; i++) {
    BandSubgraphPreparationType type = (BandSubgraphPreparationType)i;
    if (strncmp(BandSubgraphPreparationToString(type), name,
                strlen(BandSubgraphPreparationToString(type))) == 0) {
      return type;
    }
  }
  return kBandNumSubgraphPreparationType;
}

const char* BandDataTypeToString(BandDataType type) {
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
    default: {
    }
  }
  return "Unknown type";
}

const char* BandDeviceToString(BandDeviceFlag flag) {
  switch (flag) {
    case kBandCPU:
      return "CPU";
    case kBandGPU:
      return "GPU";
    case kBandDSP:
      return "DSP";
    case kBandNPU:
      return "NPU";
    default: {
    }
  }
  return "Unknown type";
}

BandDeviceFlag BandDeviceGetFlag(const char* name) {
  for (int i = 0; i < kBandNumDeviceFlag; i++) {
    BandDeviceFlag flag = (BandDeviceFlag)i;
    if (strncmp(BandDeviceToString(flag), name,
                strlen(BandDeviceToString(flag))) == 0) {
      return flag;
    }
  }
  return kBandNumDeviceFlag;
}

const char* BandBackendToString(BandBackendType flag) {
  switch (flag) {
    case kBandTfLite:
      return "Tensorflow Lite";
    default: {
    }
  }
  return "Unknown type";
}

const BandBackendType BandBackendGetType(const char* name) {
  for (int i = 0; i < kBandNumBackendType; i++) {
    BandBackendType flag = (BandBackendType)i;
    if (strcmp(BandBackendToString(flag), name) == 0) {
      return flag;
    }
  }
  return kBandNumBackendType;
}

const char* BandStatusToString(BandStatus status) {
  switch (status) {
    case kBandOk:
      return "Ok";
    case kBandDelegateError:
      return "DelegateError";
    case kBandError:
      return "Error";
    default: {
    }
  }
  return "Unknown type";
}