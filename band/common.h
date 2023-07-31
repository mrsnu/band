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

// Empty template.
template <typename EnumType>
size_t EnumLength() {
  assert(false && "EnumLength is not implemented for this type.");
  return 0;
}

template <typename EnumType>
std::string ToString(EnumType t) {
  assert(false && "ToString is not implemented for this type.");
  return "";
}

template <typename EnumType>
EnumType FromString(std::string str) {
  for (size_t i = 0; i < EnumLength<EnumType>(); i++) {
    EnumType t = static_cast<EnumType>(i);
    if (ToString(t) == str) {
      return t;
    }
  }
  // fallback to the first enum value
  return static_cast<EnumType>(0);
}

enum class BackendType : size_t {
  kTfLite = 0,
};

enum class SchedulerType : size_t {
  kFixedWorker = 0,
  kRoundRobin = 1,
  kShortestExpectedLatency = 2,
  kFixedWorkerGlobalQueue = 3,
  kHeterogeneousEarliestFinishTime = 4,
  kLeastSlackTimeFirst = 5,
  kHeterogeneousEarliestFinishTimeReserved = 6,
};

enum class CPUMaskFlag : size_t {
  kAll = 0,
  kLittle = 1,
  kBig = 2,
  kPrimary = 3,
};

enum class SubgraphPreparationType : size_t {
  kNoFallbackSubgraph = 0,
  kFallbackPerWorker = 1,
  kUnitSubgraph = 2,
  kMergeUnitSubgraph = 3,
};

enum class DataType : size_t {
  kNoType = 0,
  kFloat32 = 1,
  kInt32 = 2,
  kUInt8 = 3,
  kInt64 = 4,
  kString = 5,
  kBool = 6,
  kInt16 = 7,
  kComplex64 = 8,
  kInt8 = 9,
  kFloat16 = 10,
  kFloat64 = 11,
};

size_t GetDataTypeBytes(DataType type);

enum class BufferFormat : size_t {
  // image format
  kGrayScale = 0,
  kRGB = 1,
  kRGBA = 2,
  kYV12 = 3,
  kYV21 = 4,
  kNV21 = 5,
  kNV12 = 6,
  // raw format, from tensor
  // internal format follows DataType
  kRaw = 7
};

// Buffer content orientation follows EXIF specification. The name of
// each enum value defines the position of the 0th row and the 0th column of
// the image content. See http://jpegclub.org/exif_orientation.html for
// details.
enum class BufferOrientation : size_t {
  kTopLeft = 1,
  kTopRight = 2,
  kBottomRight = 3,
  kBottomLeft = 4,
  kLeftTop = 5,
  kRightTop = 6,
  kRightBottom = 7,
  kLeftBottom = 8,
};

enum class DeviceFlag : size_t {
  kCPU = 0,
  kGPU = 1,
  kDSP = 2,
  kNPU = 3,
};

enum class QuantizationType : size_t {
  kNoQuantization = 0,
  kAffineQuantization = 1,
};

enum class WorkerType : size_t {
  kDeviceQueue = 1,
  kGlobalQueue = 2,
};

enum class JobStatus : size_t {
  kEnqueueFailed,
  kQueued,
  kSuccess,
  kSLOViolation,
  kInputCopyFailure,
  kOutputCopyFailure,
  kInvokeFailure
};

template <>
size_t EnumLength<BackendType>();
template <>
size_t EnumLength<SchedulerType>();
template <>
size_t EnumLength<CPUMaskFlag>();
template <>
size_t EnumLength<SubgraphPreparationType>();
template <>
size_t EnumLength<DataType>();
template <>
size_t EnumLength<BufferFormat>();
template <>
size_t EnumLength<BufferOrientation>();
template <>
size_t EnumLength<DeviceFlag>();
template <>
size_t EnumLength<QuantizationType>();

template <>
std::string ToString(BackendType backend_type);
template <>
std::string ToString(SchedulerType scheduler_type);
template <>
std::string ToString(CPUMaskFlag cpu_mask_flag);
template <>
std::string ToString(SubgraphPreparationType subgraph_preparation_type);
template <>
std::string ToString(DataType data_type);
template <>
std::string ToString(BufferFormat buffer_format);
template <>
std::string ToString(BufferOrientation buffer_orientation);
template <>
std::string ToString(DeviceFlag device_flag);
template <>
std::string ToString(QuantizationType);
template <>
std::string ToString(JobStatus job_status);

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
  JobStatus status = JobStatus::kQueued;
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

// Helper macro to return error status
#define RETURN_IF_ERROR(expr) \
  {                           \
    auto status = (expr);     \
    if (!status.ok()) {       \
      return status;          \
    }                         \
  }

#endif  // BAND_COMMON_H_