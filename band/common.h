/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#include <array>
#include <vector>
#include <deque>

namespace band {
typedef int WorkerId;
typedef int ModelId;
typedef int JobId;
typedef int GraphJobId;

struct BitMask {
  BitMask() : mask() { mask.fill(false); }
  BitMask(const std::array<bool, 256>& mask) : mask(mask) {}
  BitMask(const BitMask& other) : mask(other.mask) {}
  BitMask& operator=(const BitMask& other) {
    mask = other.mask;
    return *this;
  }
  BitMask& operator=(const std::array<bool, 256>& other) {
    mask = other;
    return *this;
  }
  bool operator==(const BitMask& other) const { return mask == other.mask; }
  bool operator!=(const BitMask& other) const { return mask != other.mask; }
  bool operator<(const BitMask& other) const {
    for (int i = 0; i < 256; i++) {
      if (mask[i] != other.mask[i]) {
        return mask[i] < other.mask[i];
      }
    }
    return false;
  }

  BitMask operator^(const BitMask& other) const {
    BitMask result;
    for (int i = 0; i < 256; i++) {
      result.mask[i] = mask[i] ^ other.mask[i];
    }
    return result;
  }
  
  BitMask operator&(const BitMask& other) const {
    BitMask result;
    for (int i = 0; i < 256; i++) {
      result.mask[i] = mask[i] & other.mask[i];
    }
    return result;
  }

  BitMask operator|(const BitMask& other) const {
    BitMask result;
    for (int i = 0; i < 256; i++) {
      result.mask[i] = mask[i] | other.mask[i];
    }
    return result;
  }

  BitMask operator~() const {
    BitMask result;
    for (int i = 0; i < 256; i++) {
      result.mask[i] = !mask[i];
    }
    return result;
  }

  BitMask& operator^=(const BitMask& other) {
    for (int i = 0; i < 256; i++) {
      mask[i] ^= other.mask[i];
    }
    return *this;
  }

  BitMask& operator&=(const BitMask& other) {
    for (int i = 0; i < 256; i++) {
      mask[i] &= other.mask[i];
    }
    return *this;
  }

  BitMask& operator|=(const BitMask& other) {
    for (int i = 0; i < 256; i++) {
      mask[i] |= other.mask[i];
    }
    return *this;
  }

  BitMask& operator|=(int i) {
    mask[i] = true;
    return *this;
  }

  BitMask& operator~() {
    for (int i = 0; i < 256; i++) {
      mask[i] = !mask[i];
    }
    return *this;
  }

  bool operator[](int i) const { return mask[i]; }
  bool any() const { 
    for (int i = 0; i < 256; i++) {
      if (mask[i]) {
        return true;
      }
    }
    return false;
  }
  bool all() const {
    for (int i = 0; i < 256; i++) {
      if (!mask[i]) {
        return false;
      }
    }
    return true;
  }
  bool none() const {
    for (int i = 0; i < 256; i++) {
      if (mask[i]) {
        return false;
      }
    }
    return true;
  }
  bool test(int i) const { return mask[i]; }
  void set(int i) { mask[i] = true; }
  void clear() { mask.fill(false); }
  size_t size() const { return 256; }
  std::array<bool, 256> get() const { return mask; }
  std::string ToString() const {
    std::string result;
    for (int i = 0; i < 256; i++) {
      result += mask[i] ? "1" : "0";
    }
    return result;
  }

private:
  std::array<bool, 256> mask;
};

// Empty template.
template <typename EnumType>
size_t EnumLength() {
  assert(false && "EnumLength is not implemented for this type.");
  return 0;
}

template <typename EnumType>
const char* ToString(EnumType t) {
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
  kRoundRobin,
  kRoundRobinIdle,
  kShortestExpectedLatency,
  kFixedWorkerGlobalQueue,
  kHeterogeneousEarliestFinishTime,
  kLeastSlackTimeFirst,
  kHeterogeneousEarliestFinishTimeReserved,
  kThermal,
};

enum class CPUMaskFlag : size_t {
  kAll = 0,
  kLittle,
  kBig,
  kPrimary,
};

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

enum class SensorFlag : size_t {
  kCPU = 0,
  kGPU = 1,
  kDSP = 2,
  kNPU = 3,
  kTarget = 4,
};

enum class QuantizationType : size_t {
  kNoQuantization = 0,
  kAffineQuantization,
};

enum class WorkerType : size_t {
  kDeviceQueue = 1 << 0,
  kGlobalQueue = 1 << 1,
};

enum class JobStatus : size_t {
  kEnqueueFailed = 0,
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
size_t EnumLength<SensorFlag>();
template <>
size_t EnumLength<QuantizationType>();

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
const char* ToString(SensorFlag sensor_flag);
template <>
const char* ToString(QuantizationType);
template <>
const char* ToString(JobStatus job_status);

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

  static Job CreateIdleJob(int idle_us, SubgraphKey key) {
    static int count = -1;
    Job idle_job;
    idle_job.job_id = count--;
    idle_job.is_idle_job = true;
    idle_job.idle_us = idle_us;
    idle_job.subgraph_key = key;
    return idle_job;
  }

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
  double profiled_execution_time = 0;
  double profiled_latency = 0;
  std::map<SensorFlag, double> profiled_thermal;

  // Expected
  double expected_execution_time = 0;
  double expected_latency = 0;
  std::map<SensorFlag, double> expected_thermal;

  // SLO for the job
  int64_t slo_us;

  // Profiler event_id
  size_t event_id;

  // Target worker id (only for fixed worker request)
  WorkerId target_worker_id = -1;

  // Current status for execution (Valid after planning)
  JobStatus status = JobStatus::kQueued;
  SubgraphKey subgraph_key;
  std::vector<Job> following_jobs;

  // Resolved unit subgraphs and executed subgraph keys
  BitMask resolved_unit_subgraphs;
  std::list<SubgraphKey> previous_subgraph_keys;

  // Idle job
  bool is_idle_job = false;
  int idle_us = 0;
};

// Type definition of job queue.
using JobQueue = std::deque<Job>;

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