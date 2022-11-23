/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_LITE_UTIL_H_
#define TENSORFLOW_LITE_UTIL_H_

#include <sys/stat.h>
#include <json/json.h>

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>

#include "tensorflow/lite/c/common.h"

namespace tflite {

typedef int32_t worker_id_t;
typedef int32_t thermal_t;
typedef int32_t freq_t;
typedef std::string path_t;

// data structure for identifying subgraphs within whole models
struct SubgraphKey {
  SubgraphKey() {}
  SubgraphKey(int model_id, int worker_id,
              std::set<int> input_ops, std::set<int> output_ops)
      : model_id(model_id),
        worker_id(worker_id),
        input_ops(input_ops),
        output_ops(output_ops) {}

  bool operator<(const SubgraphKey& key) const {
    if (model_id != key.model_id) {
      return model_id < key.model_id;
    }

    if (worker_id != key.worker_id) {
      return worker_id < key.worker_id;
    }

    if (input_ops != key.input_ops) {
      return input_ops < key.input_ops;
    }

    return output_ops < key.output_ops;
  }

  std::string GetInputOpsString() const;
  std::string GetOutputOpsString() const;

  int model_id;
  int worker_id;
  std::set<int> input_ops;
  std::set<int> output_ops;
  std::set<int> unit_indices;
};

using DeviceOpIndices = std::pair<TfLiteDeviceFlags, std::set<int>>;
using Tensors = std::vector<TfLiteTensor*>;

enum JobStatus {
  kTfLiteJobQueued,
  kTfLiteJobSuccess,
  kTfLiteJobSLOViolation,
  kTfLiteJobInputCopyFailure,
  kTfLiteJobOutputCopyFailure,
  kTfLiteJobInvokeFailure
};
// Job struct is the scheduling and executing unit.
// The request can specify a model by indication the model id
struct Job {
  explicit Job() : model_id(-1) {}
  explicit Job(int model_id) : model_id(model_id) {}
  explicit Job(int model_id, int64_t slo) : model_id(model_id), slo_us(slo) {}

  // For record (Valid after execution)
  int64_t enqueue_time = 0;
  int64_t invoke_time = 0;
  int64_t end_time = 0;
  int64_t latency = 0;
  int64_t estimated_latency = 0;

  int64_t slo_us = 0;
  int64_t communication_time = 0;

  std::vector<int> estimated_total_latency;
  std::vector<thermal_t> estimated_temp_diff;
  std::vector<double> estimated_ppt;

  // Constant variables (Valid after invoke)
  // TODO: better job life-cycle to change these to `const`
  int model_id;
  int input_handle = -1;
  int output_handle = -1;
  int job_id = -1;
  int sched_id = -1;
  std::string model_fname;

  // Current status for execution (Valid after planning)
  JobStatus status = kTfLiteJobQueued;
  int subgraph_idx = -1;
  int worker_id = -1;
  int device_id = -1;
  int start_unit_idx = 0;
  std::vector<Job> following_jobs;
  // see Interpreter::MakeSubgraphsForFallbackOps for details on this field
  std::set<int> resolved_tensors;
  std::list<int> previous_subgraph_indices;

  thermal_t estimated_temp;
  std::vector<thermal_t> before_temp;
  std::vector<thermal_t> after_temp;
  std::vector<thermal_t> before_target_temp;
  std::vector<thermal_t> after_target_temp;
  std::vector<freq_t> frequency;
};

// Model configuration struct.
// The configuration is given when registering the model.
struct ModelConfig {
  std::string model_fname;
  int period_ms;
  int device = -1;
  int batch_size = 1;
  int64_t slo_us = -1;
  float slo_scale = -1.f;
};

// Find model id from model name.
// If the model name is not found, return -1.
int GetModelId(std::string model_name,
               const std::map<int, ModelConfig>& model_configs);

// Find model name from model id.
// If the model id is not found, return an empty string.
std::string GetModelName(int model_id,
                         const std::map<int, ModelConfig>& model_configs);

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
inline bool FileExists(const std::string& name) {
  struct stat buffer;
  return stat(name.c_str(), &buffer) == 0;
}

// load data from the given file
// if there is no such file, then the json object will be empty
Json::Value LoadJsonObjectFromFile(std::string file_path);

// Write json object.
void WriteJsonObjectToFile(const Json::Value& json_object,
                           std::string file_path);

// The prefix of Flex op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kFlexCustomCodePrefix[] = "Flex";

// Checks whether the prefix of the custom name indicates the operation is an
// Flex operation.
bool IsFlexOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'rank' elements.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int rank, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

struct TfLiteIntArrayDeleter {
  void operator()(TfLiteIntArray* a) {
    if (a) TfLiteIntArrayFree(a);
  }
};

// Helper for Building TfLiteIntArray that is wrapped in a unique_ptr,
// So that it is automatically freed when it goes out of the scope.
std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data);

// Populates the size in bytes of a type into `bytes`. Returns kTfLiteOk for
// valid types, and kTfLiteError otherwise.
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

// Creates a stub TfLiteRegistration instance with the provided
// `custom_op_name`. The op will fail if invoked, and is useful as a
// placeholder to defer op resolution.
// Note that `custom_op_name` must remain valid for the returned op's lifetime..
TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name);

// Checks whether the provided op is an unresolved custom op.
bool IsUnresolvedCustomOp(const TfLiteRegistration& registration);

// Returns a descriptive name with the given op TfLiteRegistration.
std::string GetOpNameByRegistration(const TfLiteRegistration& registration);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_UTIL_H_
