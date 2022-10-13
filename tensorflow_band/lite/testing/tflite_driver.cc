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
#include "tensorflow/lite/testing/tflite_driver.h"

#include <algorithm>
#include <complex>
#include <memory>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/lite/builtin_op_data.h"

// NOTE: flex:delegate is removed for simple testing
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/hashtable/hashtable_ops.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace testing {

namespace {
const double kRelativeThreshold = 1e-2f;
const double kAbsoluteThreshold = 1e-4f;

// For quantized tests, we use a different error measurement from float ones.
// Assumes the baseline is a always a float TF model.
// Error of a quantized model compared to the baseline comes from two sources:
//   1. the math done with quantized inputs, and
//   2. quantization of the output.
// Assumes there is no error introduced by source 1, the theoretical maximum
// error allowed for the output is 0.5 * scale, because scale is equal to the
// size of the quantization bucket.
//
// As a result, we use `scale` as a unit for measuring the quantization error.
// To add the error introduced by source 1 as well, we need to relax the
// multiplier from 0.5 to a larger number, which is model/op dependent.
// The number below is good enough to account for both the two sources of error
// for most quantized op tests to pass.
const int kQuantizationErrorMultiplier = 4;

// Returns the value in the given position in a tensor.
template <typename T>
T Value(void* data, int index) {
  return static_cast<T*>(data)[index];
}

template <typename T>
void SetTensorData(const std::vector<T>& values, void* data) {
  T* input_ptr = static_cast<T*>(data);
  std::copy(values.begin(), values.end(), input_ptr);
}

// Implement type erasure with unique_ptr with custom deleter
using unique_void_ptr = std::unique_ptr<void, void (*)(void*)>;

template <typename T>
unique_void_ptr make_type_erased_array(size_t size) {
  return unique_void_ptr(static_cast<void*>(new T[size]),
                         [](void* data) { delete[] static_cast<T*>(data); });
}

bool IsQuantized(const TfLiteTensor& tensor) {
  if (tensor.type != kTfLiteInt8) return false;

  if (tensor.quantization.params != nullptr) {
    auto* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
    if (quantization->scale != nullptr && quantization->scale->size == 1 &&
        quantization->zero_point != nullptr &&
        quantization->zero_point->size == 1) {
      return true;
    }
  }
  return false;
}
}  // namespace

class TfLiteDriver::DataExpectation {
 public:
  DataExpectation(double relative_threshold, double absolute_threshold,
                  int quantization_error_multiplier)
      : data_(nullptr, nullptr),
        num_elements_(0),
        relative_threshold_(relative_threshold),
        absolute_threshold_(absolute_threshold),
        quantization_error_multiplier_(quantization_error_multiplier) {}

  template <typename T>
  void SetData(const string& csv_values) {
    const auto& values = testing::Split<T>(csv_values, ",");
    num_elements_ = values.size();
    data_ = make_type_erased_array<T>(num_elements_);
    SetTensorData(values, data_.get());
  }

  bool Check(bool verbose, const TfLiteTensor& tensor);

 private:
  bool CompareTwoValuesHelper(float v1, float v2) {
    float diff = std::abs(v1 - v2);
    bool error_is_large = false;
    // For very small numbers, try absolute error, otherwise go with
    // relative.
    if (std::abs(v2) < relative_threshold_) {
      error_is_large = (diff > absolute_threshold_);
    } else {
      error_is_large = (diff > relative_threshold_ * std::abs(v2));
    }
    return error_is_large;
  }

  bool CompareTwoValues(std::complex<float> v1, std::complex<float> v2) {
    return CompareTwoValues(v1.real(), v2.real()) ||
           CompareTwoValues(v1.imag(), v2.imag());
  }

  bool CompareTwoValues(float v1, float v2) {
    return CompareTwoValuesHelper(v1, v2);
  }

  template <typename T, typename TS>
  bool TypedCheck(bool verbose, const TfLiteTensor& tensor) {
    size_t tensor_size = tensor.bytes / sizeof(T);

    if (tensor_size != num_elements_) {
      std::cerr << "Expected a tensor with " << num_elements_
                << " elements, got " << tensor_size << std::endl;
      std::cerr << "while checking tensor " << tensor.name << std::endl;
      return false;
    }

    bool good_output = true;
    for (int i = 0; i < tensor_size; ++i) {
      TS computed = Value<T>(tensor.data.raw, i);
      TS reference = Value<T>(data_.get(), i);
      if (CompareTwoValues(computed, reference)) {
        good_output = false;
        if (verbose) {
          std::cerr << "  index " << i << ": got " << computed
                    << ", but expected " << reference << std::endl;
        }
      }
    }
    return good_output;
  }

  bool TypedCheckString(bool verbose, const TfLiteTensor& tensor);
  bool QuantizedCheck(bool verbose, const TfLiteTensor& tensor);

  unique_void_ptr data_;
  size_t num_elements_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

class TfLiteDriver::ShapeExpectation {
 public:
  explicit ShapeExpectation(const string& csv_values)
      : shape_(testing::Split<int32_t>(csv_values, ",")) {}

  bool CheckShape(bool verbose, const TfLiteTensor& tensor) {
    bool valid = true;
    if (tensor.dims->size == shape_.size()) {
      for (int i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != tensor.dims->data[i]) {
          valid = false;
        }
      }
    } else {
      valid = false;
    }
    if (!valid && verbose) {
      std::cerr << "Incorrect output shape while checking tensor "
                << tensor.name << std::endl;
      std::cerr << "TFLite output shape: ";
      for (int i = 0; i < tensor.dims->size; ++i) {
        std::cerr << tensor.dims->data[i] << ", ";
      }
      std::cerr << std::endl;
      std::cerr << "Expected output shape: ";
      for (int i = 0; i < shape_.size(); ++i) {
        std::cerr << shape_[i] << ", ";
      }
      std::cerr << std::endl;
    }
    return valid;
  }

 private:
  std::vector<int32_t> shape_;
};

template <>
void TfLiteDriver::DataExpectation::SetData<string>(const string& csv_values) {
  string s = absl::HexStringToBytes(csv_values);
  data_ = make_type_erased_array<char>(s.size());
  memcpy(data_.get(), s.data(), s.size());
}

bool TfLiteDriver::DataExpectation::TypedCheckString(
    bool verbose, const TfLiteTensor& tensor) {
  if (tensor.data.raw == nullptr) {
    if (verbose) {
      std::cerr << "  got empty string" << std::endl;
    }
    return false;
  }
  int expected_num_strings = GetStringCount(data_.get());
  int returned_num_strings = GetStringCount(&tensor);
  if (expected_num_strings != returned_num_strings) {
    if (verbose) {
      std::cerr << "  string count differ: got " << returned_num_strings
                << ", but expected " << expected_num_strings << std::endl;
    }
    return false;
  }
  for (int i = 0; i < returned_num_strings; ++i) {
    auto expected_ref = GetString(data_.get(), i);
    auto returned_ref = GetString(&tensor, i);
    if (expected_ref.len != returned_ref.len) {
      if (verbose) {
        std::cerr << "  index " << i << ": got string of size "
                  << returned_ref.len << ", but expected size "
                  << expected_ref.len << std::endl;
      }
      return false;
    }
    if (strncmp(expected_ref.str, returned_ref.str, returned_ref.len) != 0) {
      if (verbose) {
        std::cerr << "  index " << i << ": strings are different" << std::endl;
      }
      return false;
    }
  }

  return true;
}

bool TfLiteDriver::DataExpectation::QuantizedCheck(bool verbose,
                                                   const TfLiteTensor& tensor) {
  auto* quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
  const float scale = quantization->scale->data[0];
  const int32_t zero_point = quantization->zero_point->data[0];

  bool good_result = true;
  for (int i = 0; i < tensor.bytes; i++) {
    const int32_t computed = tensor.data.int8[i];
    const float dequantized =
        static_cast<float>(scale * (computed - zero_point));
    const float reference = Value<float>(data_.get(), i);
    if (std::abs(dequantized - reference) >
        quantization_error_multiplier_ * scale) {
      if (verbose) {
        std::cerr << "  index " << i << ": got " << dequantized
                  << ", but expected " << reference << std::endl;
      }
      good_result = false;
    }
  }
  return good_result;
}

bool TfLiteDriver::DataExpectation::Check(bool verbose,
                                          const TfLiteTensor& tensor) {
  if (IsQuantized(tensor)) {
    return QuantizedCheck(verbose, tensor);
  }

  switch (tensor.type) {
    case kTfLiteFloat32:
      return TypedCheck<float, float>(verbose, tensor);
    case kTfLiteInt32:
      return TypedCheck<int32_t, float>(verbose, tensor);
    case kTfLiteInt64:
      return TypedCheck<int64_t, float>(verbose, tensor);
    case kTfLiteUInt8:
      return TypedCheck<uint8_t, float>(verbose, tensor);
    case kTfLiteInt8:
      return TypedCheck<int8_t, float>(verbose, tensor);
    case kTfLiteBool:
      return TypedCheck<bool, float>(verbose, tensor);
    case kTfLiteString:
      return TypedCheckString(verbose, tensor);
    case kTfLiteComplex64:
      return TypedCheck<std::complex<float>, std::complex<float>>(verbose,
                                                                  tensor);
    default:
      fprintf(stderr, "Unsupported type %d in Check\n", tensor.type);
      return false;
  }
}

TfLiteDriver::TfLiteDriver(bool reference_kernel)
    : relative_threshold_(kRelativeThreshold),
      absolute_threshold_(kAbsoluteThreshold),
      quantization_error_multiplier_(kQuantizationErrorMultiplier) {
  if (reference_kernel) {
    resolver_.reset(new ops::builtin::BuiltinRefOpResolver);
  } else {
    resolver_.reset(new ops::builtin::BuiltinOpResolver);
    ops::builtin::BuiltinOpResolver* buildinop_resolver_ =
        reinterpret_cast<ops::builtin::BuiltinOpResolver*>(resolver_.get());
    buildinop_resolver_->AddCustom("RFFT2D",
                                   tflite::ops::custom::Register_RFFT2D());
    tflite::ops::custom::AddHashtableOps(buildinop_resolver_);
  }
}

TfLiteDriver::~TfLiteDriver() {
  for (auto t : tensors_to_deallocate_) {
    DeallocateStringTensor(t.second);
  }
}

void TfLiteDriver::ResetInterpreter(RuntimeConfig runtime_config) {
  (&interpreter_)->reset(new Interpreter(nullptr, runtime_config));
}

void TfLiteDriver::AllocateTensors(int model_id) {
  if (must_allocate_tensors_) {
    if (interpreter_->AllocateTensors(model_id) != kTfLiteOk) {
      Invalidate("Failed to allocate tensors");
      return;
    }
    ResetLSTMStateTensors();
    must_allocate_tensors_ = false;
  }
}

int TfLiteDriver::LoadModel(const string& bin_file_path) {
  if (!IsValid()) return -1;

  std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(GetFullPath(bin_file_path).c_str());
  if (!model) {
    Invalidate("Failed to mmap model " + bin_file_path);
    return -1;
  }

  int model_id = InterpreterBuilder::RegisterModel(
      *model, nullptr, *resolver_, &interpreter_, 1);

  if (!interpreter_) {
    Invalidate("Failed build interpreter");
    return -1;
  }

  must_allocate_tensors_ = true;

  return model_id;
}

void TfLiteDriver::ResetTensor(TfLiteTensor* tensor) {
  if (!IsValid()) return;
  memset(tensor->data.raw, 0, tensor->bytes);
}

void TfLiteDriver::ResetTensor(int model_id, int id) {
  ResetTensor(interpreter_->tensor(model_id, id));
}

void TfLiteDriver::ReshapeTensor(int model_id, int id, const string& csv_values) {
  if (!IsValid()) return;
  if (interpreter_->ResizeInputTensor(
          model_id, id, testing::Split<int>(csv_values, ",")) != kTfLiteOk) {
    Invalidate("Failed to resize input tensor " + std::to_string(id));
    return;
  }
  must_allocate_tensors_ = true;
}

TfLiteTensor* TfLiteDriver::AllocateInputTensor(int model_id, int input_index) {
  const int worker_id = interpreter_->GetRepresentativeWorkerId(kTfLiteCPU);
  size_t subgraph_index = interpreter_->GetSubgraphIdx(model_id, worker_id);

  TfLiteTensor* input = TfLiteTensorCreateLike(
      interpreter_->tensor(subgraph_index, interpreter_->inputs(subgraph_index)[input_index]));

  return input;
}

TfLiteTensor* TfLiteDriver::AllocateOutputTensor(int model_id, int output_index) {
  const int worker_id = interpreter_->GetRepresentativeWorkerId(kTfLiteCPU);
  size_t subgraph_index = interpreter_->GetSubgraphIdx(model_id, worker_id);

  TfLiteTensor* output = TfLiteTensorCreateLike(
      interpreter_->tensor(subgraph_index, interpreter_->outputs(subgraph_index)[output_index]));

  return output;
}

void TfLiteDriver::SetDataToTensor(TfLiteTensor* tensor, const string& csv_values) {
  if (!IsValid()) return;
  switch (tensor->type) {
    case kTfLiteFloat32: {
      const auto& values = testing::Split<float>(csv_values, ",");
      if (!CheckSizes<float>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt32: {
      const auto& values = testing::Split<int32_t>(csv_values, ",");
      if (!CheckSizes<int32_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt64: {
      const auto& values = testing::Split<int64_t>(csv_values, ",");
      if (!CheckSizes<int64_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteUInt8: {
      const auto& values = testing::Split<uint8_t>(csv_values, ",");
      if (!CheckSizes<uint8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt8: {
      const auto& values = testing::Split<int8_t>(csv_values, ",");
      if (!CheckSizes<int8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteBool: {
      const auto& values = testing::Split<bool>(csv_values, ",");
      if (!CheckSizes<bool>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetInput"));
      return;
  }
}

void TfLiteDriver::SetInput(int model_id, int id, const string& csv_values) {
  if (!IsValid()) return;
  auto* tensor = interpreter_->tensor(model_id, id);
  switch (tensor->type) {
    case kTfLiteFloat32: {
      const auto& values = testing::Split<float>(csv_values, ",");
      if (!CheckSizes<float>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt32: {
      const auto& values = testing::Split<int32_t>(csv_values, ",");
      if (!CheckSizes<int32_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt64: {
      const auto& values = testing::Split<int64_t>(csv_values, ",");
      if (!CheckSizes<int64_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteUInt8: {
      const auto& values = testing::Split<uint8_t>(csv_values, ",");
      if (!CheckSizes<uint8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt8: {
      const auto& values = testing::Split<int8_t>(csv_values, ",");
      if (!CheckSizes<int8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteBool: {
      const auto& values = testing::Split<bool>(csv_values, ",");
      if (!CheckSizes<bool>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteString: {
      string s = absl::HexStringToBytes(csv_values);

      DeallocateStringTensor(tensors_to_deallocate_[id]);
      AllocateStringTensor(id, s.size(), tensor);
      memcpy(tensor->data.raw, s.data(), s.size());

      break;
    }
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetInput"));
      return;
  }
}

void TfLiteDriver::SetThreshold(double relative_threshold,
                                double absolute_threshold) {
  relative_threshold_ = relative_threshold;
  absolute_threshold_ = absolute_threshold;
}

void TfLiteDriver::SetQuantizationErrorMultiplier(
    int quantization_error_multiplier) {
  quantization_error_multiplier_ = quantization_error_multiplier;
}

void TfLiteDriver::SetExpectation(int model_id, int id, const string& csv_values) {
  if (!IsValid()) return;
  auto* tensor = interpreter_->tensor(model_id, id);
  auto& expected_output = expected_output_[model_id];
  if (expected_output.count(id) != 0) {
    Invalidate(absl::StrCat("Overridden expectation for tensor '", id, "'"));
  }
  expected_output[id].reset(
      new DataExpectation(relative_threshold_, absolute_threshold_,
                          quantization_error_multiplier_));

  if (IsQuantized(*tensor)) {
    expected_output[id]->SetData<float>(csv_values);
    return;
  }

  switch (tensor->type) {
    case kTfLiteFloat32:
      expected_output[id]->SetData<float>(csv_values);
      break;
    case kTfLiteInt32:
      expected_output[id]->SetData<int32_t>(csv_values);
      break;
    case kTfLiteInt64:
      expected_output[id]->SetData<int64_t>(csv_values);
      break;
    case kTfLiteUInt8:
      expected_output[id]->SetData<uint8_t>(csv_values);
      break;
    case kTfLiteInt8:
      expected_output[id]->SetData<int8_t>(csv_values);
      break;
    case kTfLiteBool:
      expected_output[id]->SetData<bool>(csv_values);
      break;
    case kTfLiteString:
      expected_output[id]->SetData<string>(csv_values);
      break;
    case kTfLiteComplex64:
      expected_output[id]->SetData<std::complex<float>>(csv_values);
      break;
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetExpectation"));
      return;
  }
}

void TfLiteDriver::SetShapeExpectation(int model_id, int id, const string& csv_values) {
  if (!IsValid()) return;
  auto& expected_output_shape = expected_output_shape_[model_id];
  if (expected_output_shape.count(id) != 0) {
    Invalidate(
        absl::StrCat("Overridden shape expectation for tensor '", id, "'"));
  }
  expected_output_shape[id].reset(new ShapeExpectation(csv_values));
}

void TfLiteDriver::Invoke(int model_id) {
  if (!IsValid()) return;
  if (interpreter_->Invoke(model_id) != kTfLiteOk) {
    Invalidate("Failed to invoke interpreter");
  }
}

void TfLiteDriver::InvokeWithInput(std::vector<Job>& requests, std::vector<Tensors>& inputs,
                                        std::vector<Tensors>& outputs)  {
  if (!IsValid()) return;
  interpreter_->InvokeModelsSync(requests, inputs, outputs);
}

void TfLiteDriver::InvokeThroughPlanner(int model_id) {
  if (!IsValid()) return;
  interpreter_->InvokeModelsSync({Job(model_id)});
}

bool TfLiteDriver::CheckResults(int model_id) {
  if (!IsValid()) return false;
  bool success = true;
  for (const auto& p : expected_output_[model_id]) {
    int id = p.first;
    auto* tensor = interpreter_->tensor(model_id, id);
    if (!p.second->Check(/*verbose=*/false, *tensor)) {
      // Do not invalidate anything here. Instead, simply output the
      // differences and return false. Invalidating would prevent all
      // subsequent invocations from running..
      std::cerr << "There were errors in invocation '" << GetInvocationId()
                << "', output tensor '" << id << "':" << std::endl;
      p.second->Check(/*verbose=*/true, *tensor);
      success = false;
      SetOverallSuccess(false);
    }
  }
  for (const auto& p : expected_output_shape_[model_id]) {
    int id = p.first;
    auto* tensor = interpreter_->tensor(model_id, id);
    if (!p.second->CheckShape(/*verbose=*/false, *tensor)) {
      // Do not invalidate anything here. Instead, simply output the
      // differences and return false. Invalidating would prevent all
      // subsequent invocations from running..
      std::cerr << "There were errors in invocation '" << GetInvocationId()
                << "', output tensor '" << id << "':" << std::endl;
      p.second->CheckShape(/*verbose=*/true, *tensor);
      success = false;
      SetOverallSuccess(false);
    }
  }
  expected_output_[model_id].clear();
  return success;
}

void TfLiteDriver::ResetLSTMStateTensors() {
  interpreter_->ResetVariableTensors(0);
}

string TfLiteDriver::ReadOutput(TfLiteTensor* tensor) {
  int num_elements = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    num_elements *= tensor->dims->data[i];
  }

  switch (tensor->type) {
    case kTfLiteFloat32:
      return JoinDefault(tensor->data.f, num_elements, ",");
    case kTfLiteInt32:
      return JoinDefault(tensor->data.i32, num_elements, ",");
    case kTfLiteInt64:
      return JoinDefault(tensor->data.i64, num_elements, ",");
    case kTfLiteUInt8:
      return Join(tensor->data.uint8, num_elements, ",");
    case kTfLiteInt8:
      return Join(tensor->data.int8, num_elements, ",");
    case kTfLiteBool:
      return JoinDefault(tensor->data.b, num_elements, ",");
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::ReadOutput"));
      return "";
  }
}

string TfLiteDriver::ReadOutput(int model_id, int id) {
  return ReadOutput(interpreter_->tensor(model_id, id));
}

}  // namespace testing
}  // namespace tflite