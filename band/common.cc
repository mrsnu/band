#include "band/common.h"

#include "band/logger.h"

namespace band {

std::string ToString(BackendType backend_type) {
  switch (backend_type) {
    case BackendType::kBandTfLite: {
      return "Tensorflow Lite";
    } break;
    default: {
      return "Unknown backend type";
    }
  }
}

std::string ToString(CPUMaskFlag cpu_mask_flags) {
  switch (cpu_mask_flags) {
    case CPUMaskFlag::kBandAll: {
      return "ALL";
    } break;
    case CPUMaskFlag::kBandLittle: {
      return "LITTLE";
    } break;
    case CPUMaskFlag::kBandBig: {
      return "BIG";
    } break;
    case CPUMaskFlag::kBandPrimary: {
      return "PRIMARY";
    } break;
    default: {
      return "Unknown CPU mask flag";
    }
  }
}

std::string ToString(SchedulerType scheduler_type) {
  switch (scheduler_type) {
    case SchedulerType::kBandFixedWorker: {
      return "fixed_worker";
    } break;
    case SchedulerType::kBandRoundRobin: {
      return "round_robin";
    } break;
    case SchedulerType::kBandShortestExpectedLatency: {
      return "shortest_expected_latency";
    } break;
    case SchedulerType::kBandFixedWorkerGlobalQueue: {
      return "fixed_worker_global_queue";
    } break;
    case SchedulerType::kBandHeterogeneousEarliestFinishTime: {
      return "heterogeneous_earliest_finish_time";
    } break;
    case SchedulerType::kBandLeastSlackTimeFirst: {
      return "least_slack_time_first";
    } break;
    case SchedulerType::kBandHeterogeneousEarliestFinishTimeReserved: {
      return "heterogeneous_earliest_finish_time_reserved";
    } break;
    default : {
      return "Unknown scheduler type";
    } break;
  }
}

std::string ToString(SubgraphPreparationType subgraph_preparation_type) {
  switch (subgraph_preparation_type) {
    case SubgraphPreparationType::kBandNoFallbackSubgraph: {
      return "no_fallback_subgraph";
    } break;
    case SubgraphPreparationType::kBandFallbackPerWorker: {
      return "fallback_per_worker";
    } break;
    case SubgraphPreparationType::kBandUnitSubgraph: {
      return "unit_subgraph";
    } break;
    case SubgraphPreparationType::kBandMergeUnitSubgraph: {
      return "merge_unit_subgraph";
    } break;
    default : {
      return "Unknown subgraph preparation type";
    } break;
  }
}

std::string ToString(DataType data_type) {
  switch (data_type) {
    case DataType::kBandNoType: {
      return "NoType";
    } break;
    case DataType::kBandFloat32: {
      return "Float32";
    } break;
    case DataType::kBandInt16: {
      return "Int16";
    } break;
    case DataType::kBandInt32: {
      return "Int32";
    } break;
    case DataType::kBandUInt8: {
      return "UInt8";
    } break;
    case DataType::kBandInt8: {
      return "Int8";
    } break;
    case DataType::kBandInt64: {
      return "Int64";
    } break;
    case DataType::kBandBool: {
      return "Bool";
    } break;
    case DataType::kBandComplex64: {
      return "Complex64";
    } break;
    case DataType::kBandString: {
      return "String";
    } break;
    case DataType::kBandFloat16: {
      return "Float16";
    } break;
    case DataType::kBandFloat64: {
      return "Float64";
    } break;
    default : {
      return "Unknown data type";
    } break;
  }
}

std::string ToString(DeviceFlag device_flags) {
  switch (device_flags) {
    case DeviceFlag::kBandCPU: {
      return "CPU";
    } break;
    case DeviceFlag::kBandGPU: {
      return "GPU";
    } break;
    case DeviceFlag::kBandDSP: {
      return "DSP";
    } break;
    case DeviceFlag::kBandNPU: {
      return "NPU";
    } break;
    default : {
      return "Unknown device flag";
    } break;
  }
}

std::string ToString(JobStatus job_status) {
  switch (job_status) {
    case JobStatus::kEnqueueFailed: {
      return "EnqueueFailed";
    } break;
    case JobStatus::kQueued: {
      return "Queued";
    } break;
    case JobStatus::kSuccess: {
      return "Success";
    } break;
    case JobStatus::kSLOViolation: {
      return "SLOViolation";
    } break;
    case JobStatus::kInputCopyFailure: {
      return "InputCopyFailure";
    } break;
    case JobStatus::kOutputCopyFailure: {
      return "OutputCopyFailure";
    } break;
    case JobStatus::kInvokeFailure: {
      return "InvokeFailure";
    } break;
  }
  BAND_LOG_PROD(BAND_LOG_ERROR, "Unknown job status: %d", job_status);
  return "Unknown job status";
}

template <>
SchedulerType FromString(std::string str) {
  for (int i = 0; i < SchedulerType::kBandNumSchedulerType; i++) {
    SchedulerType type = static_cast<SchedulerType>(i);
    if (ToString(type) == str) {
      return type;
    }
  }
  BAND_LOG_PROD(BAND_LOG_ERROR,
                "Unknown scheduler type: %s. Fallback to fixed worker",
                str.c_str());
  return SchedulerType::kBandFixedWorker;
}

template <>
SubgraphPreparationType FromString(std::string str) {
  for (int i = 0; i < SubgraphPreparationType::kBandNumSubgraphPreparationType; i++) {
    SubgraphPreparationType type = static_cast<SubgraphPreparationType>(i);
    if (ToString(type) == str) {
      return type;
    }
  }
  BAND_LOG_PROD(
      BAND_LOG_ERROR,
      "Unknown subgraph preparation type: %s. Fallback to no_fallback_subgraph",
      str.c_str());
  return SubgraphPreparationType::kBandNoFallbackSubgraph;
}

template <>
DataType FromString(std::string str) {
  for (int i = 0; i < DataType::kBandNumDataType; i++) {
    DataType type = static_cast<DataType>(i);
    if (ToString(type) == str) {
      return type;
    }
  }
  BAND_LOG_PROD(BAND_LOG_ERROR, "Unknown data type: %s. Fallback to Float64",
                str.c_str());
  return DataType::kBandFloat64;
}

template <>
DeviceFlag FromString(std::string str) {
  for (int i = 0; i < kBandNumDeviceFlag; i++) {
    DeviceFlag flag = static_cast<DeviceFlag>(i);
    if (ToString(flag) == str) {
      return flag;
    }
  }
  BAND_LOG_PROD(BAND_LOG_ERROR, "Unknown device flag: %s. Fallback to CPU",
                str.c_str());
  return DeviceFlag::kBandCPU;
}

std::ostream& operator<<(std::ostream& os, const JobStatus& status) {
  switch (status) {
    case JobStatus::kEnqueueFailed: {
      return os << "EnqueueFailed";
    } break;
    case JobStatus::kQueued: {
      return os << "Queued";
    } break;
    case JobStatus::kSuccess: {
      return os << "Success";
    } break;
    case JobStatus::kSLOViolation: {
      return os << "SLOViolation";
    } break;
    case JobStatus::kInputCopyFailure: {
      return os << "InputCopyFailure";
    } break;
    case JobStatus::kOutputCopyFailure: {
      return os << "OutputCopyFailure";
    } break;
    case JobStatus::kInvokeFailure: {
      return os << "InvokeFailure";
    } break;
  }
  BAND_LOG_PROD(BAND_LOG_ERROR, "Unknown job status: %d", status);
  return os;
}

SubgraphKey::SubgraphKey() {}
// special case - entire model subgraph
SubgraphKey::SubgraphKey(ModelId model_id, WorkerId worker_id,
                         std::set<int> unit_indices_set)
    : model_id(model_id), worker_id(worker_id) {
  for (int unit_index : unit_indices_set) {
    unit_indices.set(unit_index);
  }
}

bool SubgraphKey::operator<(const SubgraphKey& key) const {
  if (model_id != key.GetModelId()) {
    return model_id < key.GetModelId();
  }

  if (worker_id != key.GetWorkerId()) {
    return worker_id < key.GetWorkerId();
  }

  return unit_indices.to_ullong() < key.unit_indices.to_ullong();
}

bool SubgraphKey::operator==(const SubgraphKey& key) const {
  return (model_id == key.GetModelId()) && (worker_id == key.GetWorkerId()) &&
         (unit_indices == key.unit_indices);
}

bool SubgraphKey::operator!=(const SubgraphKey& key) const {
  return !(*this == key);
}

std::string IndexSetToString(const std::set<int>& indices) {
  std::string result;
  if (indices.size()) {
    for (const int& index : indices) {
      result += std::to_string(index) + ",";
    }
    result.pop_back();
  }
  return result;
}

const BitMask& SubgraphKey::GetUnitIndices() const { return unit_indices; }

std::set<int> SubgraphKey::GetUnitIndicesSet() const {
  std::set<int> indices;
  for (size_t i = 0; i < unit_indices.size(); i++) {
    if (unit_indices.test(i)) {
      indices.insert(i);
    }
  }
  return indices;
}

std::string SubgraphKey::GetUnitIndicesString() const {
  return IndexSetToString(GetUnitIndicesSet());
}

std::string SubgraphKey::ToString() const {
  return "Model id " + std::to_string(model_id) + " Worker id " +
         std::to_string(worker_id) + " Unit indices (" +
         GetUnitIndicesString() + ")";
}

bool SubgraphKey::IsValid() const {
  return (model_id != -1) && (worker_id != -1);
}

std::size_t SubgraphHash::operator()(const SubgraphKey& p) const {
  auto hash_func = std::hash<int>();
  std::size_t hash = hash_func(p.GetModelId()) ^ hash_func(p.GetWorkerId());

  auto hash_set = [hash_func, &hash](const std::set<int>& set) {
    for (int e : set) hash ^= hash_func(e);
  };

  hash_set(p.GetUnitIndicesSet());

  return hash;
}

std::string Job::ToJson() const {
  return "{\"enqueue_time\":" + std::to_string(enqueue_time) +
         ",\"invoke_time\":" + std::to_string(invoke_time) +
         ",\"end_time\":" + std::to_string(end_time) +
         ",\"profiled_execution_time\":" +
         std::to_string(profiled_execution_time) +
         ",\"expected_execution_time\":" +
         std::to_string(expected_execution_time) +
         ",\"expected_latency\":" + std::to_string(expected_latency) +
         ",\"slo_us\":" + std::to_string(slo_us) +
         ",\"model_id\":" + std::to_string(model_id) +
         (model_fname != "" ? ",\"model_fname\":" + model_fname : "") +
         ",\"unit_indices\":" + subgraph_key.GetUnitIndicesString() +
         ",\"job_id\":" + std::to_string(job_id) + "}";
}

std::size_t JobIdBitMaskHash::operator()(
    const std::pair<int, BitMask>& p) const {
  auto hash_func = std::hash<int>();
  return hash_func(p.first) ^ hash_func(p.second.to_ullong());
}
}  // namespace band
