#ifndef BAND_COMMON_H_
#define BAND_COMMON_H_

#include <iostream>
#include <bitset>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "band/c/common.h"

namespace Band {

typedef int WorkerId;
typedef int ModelId;
typedef int JobId;

using BitMask = std::bitset<64>;

static size_t kBandNumCpuMasks = 4;
static size_t kBandNumSchedulerTypes = 7;
static size_t kBandNumSubgraphPreparationType = 4;
static size_t kBandNumDevices = 4;

// Empty template.
template<typename T>
T FromString(std::string str) {}

enum class BackendType : int {
  TfLite = 0
};
std::string GetName(BackendType backend_type);

enum class CPUMaskFlags : int {
  All = 0,
  Little = 1,
  Big = 2,
  Primary = 3
};
std::string GetName(CPUMaskFlags cpu_mask_flags);

enum class SchedulerType : int {
  FixedWorker = 0,
  RoundRobin = 1,
  ShortestExpectedLatency = 2,
  FixedWorkerGlobalQueue = 3,
  HeterogeneousEarliestFinishTime = 4,
  LeastSlackTimeFirst = 5,
  HeterogeneousEarliestFinishTimeReserved = 6
};
std::string GetName(SchedulerType scheduler_type);
template<>
SchedulerType FromString(std::string str);

enum class SubgraphPreparationType : int {
  NoFallbackSubgraph = 0,
  FallbackPerWorker = 1,
  UnitSubgraph = 2,
  MergeUnitSubgraph = 3
};
std::string GetName(SubgraphPreparationType subgraph_preparation_type);
template<>
SubgraphPreparationType FromString(std::string str);

enum class DataType : int {
  NoType = 0,
  Float32 = 1,
  Int32 = 2,
  UInt8 = 3,
  Int64 = 4,
  String = 5,
  Bool = 6,
  Int16 = 7,
  Complex64 = 8,
  Int8 = 9,
  Float16 = 10,
  Float64 = 11
};
std::string GetName(DataType data_type);

enum class DeviceFlags : int {
  CPU = 0,
  GPU = 1,
  DSP = 2,
  NPU = 3
};
std::string GetName(DeviceFlags device_flags);
template<>
DeviceFlags FromString(std::string str);

enum class JobStatus {
  Queued,
  Success,
  SLOViolation,
  InputCopyFailure,
  OutputCopyFailure,
  InvokeFailure
};
std::string GetName(JobStatus job_status);

// data structure for identifying subgraphs within whole models
class SubgraphKey {
 public:
  SubgraphKey();
  SubgraphKey(ModelId model_id, WorkerId worker_id,
              std::set<int> unit_indices = {});
  bool operator<(const SubgraphKey& key) const;
  bool operator==(const SubgraphKey& key) const;
  bool operator!=(const SubgraphKey& key) const;

  ModelId GetModelId() const { return model_id; }
  WorkerId GetWorkerId() const { return worker_id; }

  const BitMask& GetUnitIndices() const;
  std::set<int> GetUnitIndicesSet() const;
  std::string GetUnitIndicesString() const;

  std::string ToString() const;
  bool IsValid() const;

 private:
  ModelId model_id = -1;
  WorkerId worker_id = -1;
  BitMask unit_indices;
};

// hash function to use SubgraphKey as a key
// https://stackoverflow.com/a/32685618
struct SubgraphHash {
  std::size_t operator()(const SubgraphKey& p) const;
};

std::ostream& operator<<(std::ostream& os, const JobStatus& status);

// Job struct is the scheduling and executing unit.
// The request can specify a model by indication the model id
struct Job {
  explicit Job() : model_id(-1) {}
  explicit Job(ModelId model_id) : model_id(model_id) {}
  explicit Job(ModelId model_id, int64_t slo)
      : model_id(model_id), slo_us(slo) {}

  // For record (Valid after execution)
  int64_t enqueue_time = 0;
  int64_t invoke_time = 0;
  int64_t end_time = 0;
  // Profiled invoke execution time
  int64_t profiled_execution_time = 0;
  // Expected invoke execution time
  int64_t expected_execution_time = 0;
  // Expected total latency
  int64_t expected_latency = 0;
  int64_t slo_us = 0;

  // Target worker id (only for fixed worker request)
  WorkerId target_worker_id = -1;

  // Constant variables (Valid after invoke)
  // TODO: better job life-cycle to change these to `const`
  ModelId model_id;
  int input_handle = -1;
  int output_handle = -1;
  JobId job_id = -1;
  int sched_id = -1;
  std::string model_fname;
  bool require_callback = true;

  // Current status for execution (Valid after planning)
  JobStatus status = JobStatus::Queued;
  SubgraphKey subgraph_key;
  std::vector<Job> following_jobs;

  // Resolved unit subgraphs and executed subgraph keys
  BitMask resolved_unit_subgraphs;
  std::list<SubgraphKey> previous_subgraph_keys;
};

// hash function to use pair<int, BitMask> as map key in cache_
// https://stackoverflow.com/a/32685618
struct CacheHash {
  std::size_t operator()(const std::pair<int, BitMask>& p) const;
};

}  // namespace Band

#endif  // BAND_COMMON_H_