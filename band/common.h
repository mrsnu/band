#ifndef BAND_COMMON_H_
#define BAND_COMMON_H_

#include <bitset>
#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>


namespace band {
typedef int WorkerId;
typedef int ModelId;
typedef int JobId;

using BitMask = std::bitset<64>;

enum class SchedulerType : size_t;
enum class SubgraphPreparationType : size_t;
enum class DataType : size_t;
enum class DeviceFlags : size_t;

// Empty template.
template <typename T>
T FromString(std::string str) {
  assert(false && "FromString not implemented for this type.");
}

template <typename T>
size_t GetSize() {
  assert(false && "GetSize not implemented for this type.");
  return -1;
}

enum class BackendType : size_t { TfLite = 0 };
std::string GetName(BackendType backend_type);

// NOTE: Please update the GetSize() function when adding a new scheduler type.
enum class CPUMaskFlags : size_t { All = 0, Little = 1, Big = 2, Primary = 3 };
template <>
size_t GetSize<CPUMaskFlags>();
std::string GetName(CPUMaskFlags cpu_mask_flags);

// NOTE: Please update the GetSize() function when adding a new scheduler type.
enum class SchedulerType : size_t {
  FixedWorker = 0,
  RoundRobin = 1,
  ShortestExpectedLatency = 2,
  FixedWorkerGlobalQueue = 3,
  HeterogeneousEarliestFinishTime = 4,
  LeastSlackTimeFirst = 5,
  HeterogeneousEarliestFinishTimeReserved = 6
};
template <>
size_t GetSize<SchedulerType>();
std::string GetName(SchedulerType scheduler_type);
template <>
SchedulerType FromString(std::string str);

// NOTE: Please update the GetSize() function when adding a new scheduler type.
enum class SubgraphPreparationType : size_t {
  NoFallbackSubgraph = 0,
  FallbackPerWorker = 1,
  UnitSubgraph = 2,
  MergeUnitSubgraph = 3
};
template <>
size_t GetSize<SubgraphPreparationType>();
std::string GetName(SubgraphPreparationType subgraph_preparation_type);
template <>
SubgraphPreparationType FromString(std::string str);

// NOTE: Please update the GetSize() function when adding a new scheduler type.
enum class DataType : size_t {
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
template <>
size_t GetSize<DataType>();
std::string GetName(DataType data_type);
template <>
DataType FromString(std::string str);

// NOTE: Please update the GetSize() function when adding a new scheduler type.
enum class DeviceFlags : size_t { CPU = 0, GPU = 1, DSP = 2, NPU = 3 };
template <>
size_t GetSize<DeviceFlags>();
std::string GetName(DeviceFlags device_flags);
template <>
DeviceFlags FromString(std::string str);

enum class WorkerType : size_t {
  DeviceQueue = 1,
  GlobalQueue = 2,
};

enum class JobStatus : size_t {
  Queued,
  Success,
  SLOViolation,
  InputCopyFailure,
  OutputCopyFailure,
  InvokeFailure
};
std::string GetName(JobStatus job_status);

enum class QuantizationType {
  NoQuantization = 0,
  AffineQuantization = 1,
};
std::string GetName(QuantizationType);

struct AffineQuantizationParams {
  std::vector<float> scale;
  std::vector<int32_t> zero_point;
  int32_t quantized_dimension;
};

class Quantization {
 public:
  Quantization(QuantizationType type, void* params)
      : type_(type), params_(params) {}
  QuantizationType GetType() { return type_; }
  void* GetParams() { return params_; }
  void SetParams(void* params) { params_ = params; }

 private:
  QuantizationType type_;
  void* params_;
};

// Optional parameters for model request
// `target_worker`: designate the target worker for a request.
// [default : -1 (not specified)] This option requires the FixedWorkerScheduler.
// `require_callback`: report if OnEndRequest is specified in an engine
// [default: true]
// `slo_us` and `slo_scale`: specifying an SLO value for a model.
// Setting `slo_scale` will make the SLO =  slo_scale * profiled latency of
// that model. `slo_scale` will be ignored if `slo_us` is given
// (i.e., no reason to specify both options). [default : -1 (not specified)]
struct RequestOption {
  int target_worker;
  bool require_callback;
  int slo_us;
  float slo_scale;

  static RequestOption GetDefaultOption() { return {-1, true, -1, -1.f}; }
};

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

  std::string ToJson() const;

  // Constant variables (Valid after invoke)
  // TODO: better job life-cycle to change these to `const`
  ModelId model_id;
  int input_handle = -1;
  int output_handle = -1;
  JobId job_id = -1;
  int sched_id = -1;
  std::string model_fname;
  bool require_callback = true;

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
  int64_t slo_us;

  // Target worker id (only for fixed worker request)
  WorkerId target_worker_id = -1;

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
struct JobIdBitMaskHash {
  std::size_t operator()(const std::pair<int, BitMask>& p) const;
};

}  // namespace band

#endif  // BAND_COMMON_H_