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
typedef int CallbackId;

using BitMask = std::bitset<64>;

// Empty template.
/**
 * @brief Returns the length of an enumeration type.
 * 
 * This function returns the number of elements in an enumeration type.
 * If the function is called for an unsupported enumeration type, it will assert and return 0.
 * 
 * @tparam EnumType The enumeration type.
 * @return The number of elements in the enumeration type.
 */
template <typename EnumType>
size_t EnumLength() {
  assert(false && "EnumLength is not implemented for this type.");
  return 0;
}

/**
 * Converts an enumeration value to its corresponding string representation.
 * 
 * @tparam EnumType The enumeration type.
 * @param t The enumeration value to convert.
 * @return The string representation of the enumeration value.
 */
template <typename EnumType>
const char* ToString(EnumType t) {
  assert(false && "ToString is not implemented for this type.");
  return "";
}

/**
 * Converts a string representation of an enumeration value to the corresponding enumeration value.
 * 
 * @tparam EnumType The enumeration type.
 * @param str The string representation of the enumeration value.
 * @return The corresponding enumeration value.
 * 
 * @note This function assumes that the enumeration type has a valid ToString() function defined.
 * If the string representation does not match any enumeration value, the function will fallback to the first enum value.
 */
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

// disable the warning of enum class to size_t
enum class LogSeverity : size_t {
  // for internal use
  kInternal = 0,
  // for general information (default)
  kInfo,
  kWarning,
  kError
};

// Backend type for the model
enum class BackendType : size_t {
  kTfLite = 0,
};

// Scheduler type for the model
enum class SchedulerType : size_t {
  kFixedWorker = 0,
  kRoundRobin,
  kShortestExpectedLatency,
  kFixedWorkerGlobalQueue,
  kHeterogeneousEarliestFinishTime,
  kLeastSlackTimeFirst,
  kHeterogeneousEarliestFinishTimeReserved,
};

// CPU mask flag for the model
enum class CPUMaskFlag : size_t {
  kAll = 0,
  kLittle,
  kBig,
  kPrimary,
};

// Subgraph preparation type for the model
enum class SubgraphPreparationType : size_t {
  kNoFallbackSubgraph = 0,
  kFallbackPerWorker,
  kUnitSubgraph,
  kMergeUnitSubgraph,
};

enum class DataType : size_t {
  kNoType = 0,
  kFloat32,
  kInt32,
  kUInt8,
  kInt64,
  kString,
  kBool,
  kInt16,
  kComplex64,
  kInt8,
  kFloat16,
  kFloat64,
};

size_t GetDataTypeBytes(DataType type);

enum class BufferFormat : size_t {
  // image format
  kGrayScale = 0,
  kRGB,
  kRGBA,
  kYV12,
  kYV21,
  kNV21,
  kNV12,
  // raw format, from tensor
  // internal format follows DataType
  kRaw
};

// Buffer content orientation follows EXIF specification. The name of
// each enum value defines the position of the 0th row and the 0th column of
// the image content. See http://jpegclub.org/exif_orientation.html for
// details.
// 图像数据在缓冲区中的存储方向遵守 EXIF 规范，其中每种枚举类型的名称都指定了图像内容的起始行（第 0 行）和起始列（第 0 列）的位置。
// 具体细节可以参考 http://jpegclub.org/exif_orientation.html。
enum class BufferOrientation : size_t {
  kTopLeft = 1,
  kTopRight,
  kBottomRight,
  kBottomLeft,
  kLeftTop,
  kRightTop,
  kRightBottom,
  kLeftBottom,
};

enum class DeviceFlag : size_t {
  kCPU = 0,
  kGPU,
  kDSP,
  kNPU,
};

// Quantization type for the model
enum class QuantizationType : size_t {
  kNoQuantization = 0,
  kAffineQuantization,
  // 采用 仿射量化（Affine Quantization）的方式
};

// Worker type for the model
enum class WorkerType : size_t {
  kDeviceQueue = 1 << 0,
  // 按照设备队列
  kGlobalQueue = 1 << 1,
  // 按照全局队列，共享队列
};

enum class JobStatus : size_t {
  kEnqueueFailed = 0,
  // 入队失败
  kQueued,
  // 已入队
  kSuccess,
  // 成功
  kSLOViolation,
  // SLO 违反
  kInputCopyFailure,
  // 输入拷贝失败
  kOutputCopyFailure,
  // 输出拷贝失败
  kInvokeFailure
  // 调用失败
};

template <>
size_t EnumLength<LogSeverity>();
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
const char* ToString(LogSeverity log_severity);
template <>
const char* ToString(BackendType backend_type);
template <>
const char* ToString(SchedulerType scheduler_type);
template <>
const char* ToString(CPUMaskFlag cpu_mask_flag);
template <>
const char* ToString(SubgraphPreparationType subgraph_preparation_type);
template <>
const char* ToString(DataType data_type);
template <>
const char* ToString(BufferFormat buffer_format);
template <>
const char* ToString(BufferOrientation buffer_orientation);
template <>
const char* ToString(DeviceFlag device_flag);
template <>
const char* ToString(QuantizationType);
template <>
const char* ToString(JobStatus job_status);

struct AffineQuantizationParams {
  std::vector<float> scale;
  // 缩放因子
  std::vector<int32_t> zero_point;
  // 表示量化零点的整数 允许量化后的整数值能表示正负
  int32_t quantized_dimension;
  // 量化维度
};

class Quantization {
  // 管理量化过程中的不同类型和参数
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
// 模型请求的一些可选参数包括：
// - `target_worker`：指定处理该请求的目标工作器。如果不特别指定，默认为 -1，这意味着没有指定目标工作器。使用这个选项需要启用 FixedWorkerScheduler。
// - `require_callback`：指明是否需要在模型的引擎中指定结束请求的回调函数（OnEndRequest）。默认情况下，这个选项是开启的（true）。
// - 对于 `slo_us`（服务等级目标的微秒值）和 `slo_scale`（服务等级目标的缩放因子）：它们用于为模型设置一个服务等级目标（SLO）。通过设置 `slo_scale`，SLO 将等于模型的预测延迟乘以 `slo_scale`。
// 如果已经提供了 `slo_us`，那么 `slo_scale` 将被忽略，因为没有必要同时设置这两个参数。默认情况下，这两个参数都是未指定的（-1）。
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
  // 返回单元索引的位掩码
  std::set<int> GetUnitIndicesSet() const;
  // 返回单元索引的集合
  std::string GetUnitIndicesString() const;
  // 返回单元索引的字符串表示

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
// 重载运算符 << 以输出 JobStatus 类型的对象

// Job struct is the scheduling and executing unit.
// The request can specify a model by indication the model id
// Job 结构体是进行任务调度和执行的基本单元。通过指定模型的 ID，请求可以明确指出需要使用的模型。
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
  // 被测量的执行时间
  int64_t profiled_execution_time = 0;
  // Expected invoke execution time
  // 预期执行时间
  int64_t expected_execution_time = 0;
  // Expected total latency
  int64_t expected_latency = 0;
  int64_t slo_us;

  // Target worker id (only for fixed worker request)
  // 目标工作县线程ip
  WorkerId target_worker_id = -1;

  // Current status for execution (Valid after planning)
  JobStatus status = JobStatus::kQueued;
  // 任务的相关子图索引
  SubgraphKey subgraph_key;
  // 后续任务列表
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