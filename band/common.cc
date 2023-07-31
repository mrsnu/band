#include "band/common.h"

#include "band/logger.h"
#include "common.h"

namespace band {

template <>
size_t EnumLength<BackendType>() {
  return static_cast<size_t>(BackendType::kTfLite) + 1;
}

template <>
size_t EnumLength<SchedulerType>() {
  return static_cast<size_t>(
             SchedulerType::kHeterogeneousEarliestFinishTimeReserved) +
         1;
}

template <>
size_t EnumLength<CPUMaskFlag>() {
  return static_cast<size_t>(CPUMaskFlag::kPrimary) + 1;
}

template <>
size_t EnumLength<SubgraphPreparationType>() {
  return static_cast<size_t>(SubgraphPreparationType::kMergeUnitSubgraph) + 1;
}

template <>
size_t EnumLength<DataType>() {
  return static_cast<size_t>(DataType::kFloat64) + 1;
}

template <>
size_t EnumLength<BufferFormat>() {
  return static_cast<size_t>(BufferFormat::kRaw) + 1;
}

template <>
size_t EnumLength<BufferOrientation>() {
  return static_cast<size_t>(BufferOrientation::kLeftBottom) + 1;
}

template <>
size_t EnumLength<DeviceFlag>() {
  return static_cast<size_t>(DeviceFlag::kNPU) + 1;
}

template <>
size_t EnumLength<QuantizationType>() {
  return static_cast<size_t>(QuantizationType::kAffineQuantization) + 1;
}

template <>
std::string ToString(BackendType backend_type) {
  switch (backend_type) {
    case BackendType::kTfLite: {
      return "Tensorflow Lite";
    } break;
    default: {
      return "Unknown backend type";
    }
  }
}

template <>
std::string ToString(CPUMaskFlag cpu_mask_flag) {
  switch (cpu_mask_flag) {
    case CPUMaskFlag::kAll: {
      return "ALL";
    } break;
    case CPUMaskFlag::kLittle: {
      return "LITTLE";
    } break;
    case CPUMaskFlag::kBig: {
      return "BIG";
    } break;
    case CPUMaskFlag::kPrimary: {
      return "PRIMARY";
    } break;
    default: {
      return "Unknown CPU mask flag";
    }
  }
}

template <>
std::string ToString(SchedulerType scheduler_type) {
  switch (scheduler_type) {
    case SchedulerType::kFixedWorker: {
      return "fixed_worker";
    } break;
    case SchedulerType::kRoundRobin: {
      return "round_robin";
    } break;
    case SchedulerType::kShortestExpectedLatency: {
      return "shortest_expected_latency";
    } break;
    case SchedulerType::kFixedWorkerGlobalQueue: {
      return "fixed_worker_global_queue";
    } break;
    case SchedulerType::kHeterogeneousEarliestFinishTime: {
      return "heterogeneous_earliest_finish_time";
    } break;
    case SchedulerType::kLeastSlackTimeFirst: {
      return "least_slack_time_first";
    } break;
    case SchedulerType::kHeterogeneousEarliestFinishTimeReserved: {
      return "heterogeneous_earliest_finish_time_reserved";
    } break;
    default: {
      return "Unknown scheduler type";
    } break;
  }
}

template <>
std::string ToString(SubgraphPreparationType subgraph_preparation_type) {
  switch (subgraph_preparation_type) {
    case SubgraphPreparationType::kNoFallbackSubgraph: {
      return "no_fallback_subgraph";
    } break;
    case SubgraphPreparationType::kFallbackPerWorker: {
      return "fallback_per_worker";
    } break;
    case SubgraphPreparationType::kUnitSubgraph: {
      return "unit_subgraph";
    } break;
    case SubgraphPreparationType::kMergeUnitSubgraph: {
      return "merge_unit_subgraph";
    } break;
    default: {
      return "Unknown subgraph preparation type";
    } break;
  }
}

template <>
std::string ToString(DataType data_type) {
  switch (data_type) {
    case DataType::kNoType: {
      return "NoType";
    } break;
    case DataType::kFloat32: {
      return "Float32";
    } break;
    case DataType::kInt16: {
      return "Int16";
    } break;
    case DataType::kInt32: {
      return "Int32";
    } break;
    case DataType::kUInt8: {
      return "UInt8";
    } break;
    case DataType::kInt8: {
      return "Int8";
    } break;
    case DataType::kInt64: {
      return "Int64";
    } break;
    case DataType::kBool: {
      return "Bool";
    } break;
    case DataType::kComplex64: {
      return "Complex64";
    } break;
    case DataType::kString: {
      return "String";
    } break;
    case DataType::kFloat16: {
      return "Float16";
    } break;
    case DataType::kFloat64: {
      return "Float64";
    } break;
    default: {
      return "Unknown data type";
    } break;
  }
}

template <>
std::string ToString(DeviceFlag device_flag) {
  switch (device_flag) {
    case DeviceFlag::kCPU: {
      return "CPU";
    } break;
    case DeviceFlag::kGPU: {
      return "GPU";
    } break;
    case DeviceFlag::kDSP: {
      return "DSP";
    } break;
    case DeviceFlag::kNPU: {
      return "NPU";
    } break;
    default: {
      return "Unknown device flag";
    } break;
  }
}

template <>
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
std::string ToString(BufferFormat format_type) {
  switch (format_type) {
    case BufferFormat::kGrayScale: {
      return "GrayScale";
    } break;
    case BufferFormat::kRGB: {
      return "RGB";
    } break;
    case BufferFormat::kRGBA: {
      return "RGBA";
    } break;
    case BufferFormat::kYV12: {
      return "YV12";
    } break;
    case BufferFormat::kYV21: {
      return "YV21";
    } break;
    case BufferFormat::kNV21: {
      return "NV21";
    } break;
    case BufferFormat::kNV12: {
      return "NV12";
    } break;
    case BufferFormat::kRaw: {
      return "Raw";
    } break;
    default: {
      return "Unknown format type";
    }
  }
}

template <>
std::string ToString(BufferOrientation format_type) {
  switch (format_type) {
    case BufferOrientation::kTopLeft: {
      return "TopLeft";
    } break;
    case BufferOrientation::kTopRight: {
      return "TopRight";
    } break;
    case BufferOrientation::kBottomRight: {
      return "BottomRight";
    } break;
    case BufferOrientation::kBottomLeft: {
      return "BottomLeft";
    } break;
    case BufferOrientation::kLeftTop: {
      return "LeftTop";
    } break;
    case BufferOrientation::kRightTop: {
      return "RightTop";
    } break;
    case BufferOrientation::kRightBottom: {
      return "RightBottom";
    } break;
    case BufferOrientation::kLeftBottom: {
      return "LeftBottom";
    } break;
    default: {
      return "Unknown format type";
    } break;
  }
}

size_t GetDataTypeBytes(DataType type) {
  switch (type) {
    case DataType::kNoType:
      return 0;
    case DataType::kFloat32:
      return sizeof(float);
    case DataType::kInt32:
      return sizeof(int32_t);
    case DataType::kUInt8:
      return sizeof(uint8_t);
    case DataType::kInt8:
      return sizeof(int8_t);
    case DataType::kInt16:
      return sizeof(int16_t);
    case DataType::kInt64:
      return sizeof(int64_t);
    case DataType::kString:
      return sizeof(char);
    case DataType::kBool:
      return sizeof(bool);
    case DataType::kComplex64:
      return sizeof(double);
    case DataType::kFloat16:
      return sizeof(float) / 2;
    case DataType::kFloat64:
      return sizeof(double);
    default:
      break;
  }

  BAND_LOG_PROD(BAND_LOG_WARNING, "Unsupported data type : %s", ToString(type));
  return 0;
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
