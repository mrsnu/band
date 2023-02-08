#include "band/common.h"

namespace Band {

std::string GetName(BackendType backend_type) {
  switch (backend_type) {
    case BackendType::TfLite: {
      return "Tensorflow Lite";
    } break;
      // Note: the `default` case is deliberately not implemented to generate a
      // compiler warning for unused case.
  }
}

std::string GetName(CPUMaskFlags cpu_mask_flags) {
  switch (cpu_mask_flags) {
    case CPUMaskFlags::All: {
      return "ALL";
    } break;
    case CPUMaskFlags::Little: {
      return "LITTLE";
    } break;
    case CPUMaskFlags::Big: {
      return "BIG";
    } break;
    case CPUMaskFlags::Primary: {
      return "PRIMARY";
    } break;
  }
}

std::string GetName(SchedulerType scheduler_type) {
  switch (scheduler_type) {
    case SchedulerType::FixedWorker: {
      return "fixed_worker";
    } break;
    case SchedulerType::RoundRobin: {
      return "round_robin";
    } break;
    case SchedulerType::ShortestExpectedLatency: {
      return "shortest_expected_latency";
    } break;
    case SchedulerType::FixedWorkerGlobalQueue: {
      return "fixed_worker_global_queue";
    } break;
    case SchedulerType::HeterogeneousEarliestFinishTime: {
      return "heterogeneous_earliest_finish_time";
    } break;
    case SchedulerType::LeastSlackTimeFirst: {
      return "least_slack_time_first";
    } break;
    case SchedulerType::HeterogeneousEarliestFinishTimeReserved: {
      return "heterogeneous_earliest_finish_time_reserved";
    } break;
  }
}

SchedulerType FromString(std::string str) {
  for (int i = 0; i < kBandNumSchedulerTypes; i++) {
    SchedulerType type = static_cast<SchedulerType>(i);
    if (GetName(type) == str) {
      return type;
    }
  }
  // TODO(widiba03304): absl refactor
}

std::string GetName(SubgraphPreparationType subgraph_preparation_type) {
  switch (subgraph_preparation_type) {
    case SubgraphPreparationType::NoFallbackSubgraph: {
      return "no_fallback_subgraph";
    } break;
    case SubgraphPreparationType::FallbackPerWorker: {
      return "fallback_per_worker";
    } break;
    case SubgraphPreparationType::UnitSubgraph: {
      return "unit_subgraph";
    } break;
    case SubgraphPreparationType::MergeUnitSubgraph: {
      return "merge_unit_subgraph";
    } break;
  }
}
std::string GetName(DataType data_type) {
  switch (data_type) {
    case DataType::NoType: {
      return "NOTYPE";
    } break;
    case DataType::Float32: {
      return "FLOAT32";
    } break;
    case DataType::Int16: {
      return "INT16";
    } break;
    case DataType::Int32: {
      return "INT32";
    } break;
    case DataType::UInt8: {
      return "UINT8";
    } break;
    case DataType::Int8: {
      return "INT8";
    } break;
    case DataType::Int64: {
      return "INT64";
    } break;
    case DataType::Bool: {
      return "BOOL";
    } break;
    case DataType::Complex64: {
      return "COMPLEX64";
    } break;
    case DataType::String: {
      return "STRING";
    } break;
    case DataType::Float16: {
      return "FLOAT16";
    } break;
    case DataType::Float64: {
      return "FLOAT64";
    } break;
  }
}

std::string GetName(DeviceFlags device_flags) {
  switch (device_flags) {
    case DeviceFlags::CPU: {
      return "CPU";
    } break;
    case DeviceFlags::GPU: {
      return "GPU";
    } break;
    case DeviceFlags::DSP: {
      return "DSP";
    } break;
    case DeviceFlags::NPU: {
      return "NPU";
    } break;
  }
}

std::string GetName(JobStatus job_status) {
  switch (job_status) {
    case JobStatus::Queued: {
      return "Queued";
    } break;
    case JobStatus::Success: {
      return "Success";
    } break;
    case JobStatus::SLOViolation: {
      return "SLOViolation";
    } break;
    case JobStatus::InputCopyFailure: {
      return "InputCopyFailure";
    } break;
    case JobStatus::OutputCopyFailure: {
      return "OutputCopyFailure";
    } break;
    case JobStatus::InvokeFailure: {
      return "InvokeFailure";
    } break;
  }
}

std::ostream& operator<<(std::ostream& os, const JobStatus& status) {
  switch (status) {
    case JobStatus::Queued: {
      return os << "Queued";
    } break;
    case JobStatus::Success: {
      return os << "Success";
    } break;
    case JobStatus::SLOViolation: {
      return os << "SLOViolation";
    } break;
    case JobStatus::InputCopyFailure: {
      return os << "InputCopyFailure";
    } break;
    case JobStatus::OutputCopyFailure: {
      return os << "OutputCopyFailure";
    } break;
    case JobStatus::InvokeFailure: {
      return os << "InvokeFailure";
    } break;
  }
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

std::size_t CacheHash::operator()(const std::pair<int, BitMask>& p) const {
  auto hash_func = std::hash<int>();
  return hash_func(p.first) ^ hash_func(p.second.to_ullong());
}

}  // namespace Band