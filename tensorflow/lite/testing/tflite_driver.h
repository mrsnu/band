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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_

#include <map>
#include <memory>

// NOTE: flex:delegate is removed for simple testing
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/testing/test_runner.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace testing {

// A test runner that feeds inputs into TF Lite and verifies its outputs.
class TfLiteDriver : public TestRunner {
 public:
  /**
   * Creates a new TfLiteDriver
   * @param  delegate         The (optional) delegate to use.
   * @param  reference_kernel Whether to use the builtin reference kernel ops.
   */
  explicit TfLiteDriver(bool reference_kernel = false);
  ~TfLiteDriver() override;

  void ResetInterpreter(RuntimeConfig runtime_config) override;
  int LoadModel(const string& bin_file_path) override;
  const std::vector<int>& GetInputs(int model_id) override {
    return interpreter_->inputs(model_id);
  }
  const std::vector<int>& GetOutputs(int model_id) override {
    return interpreter_->outputs(model_id);
  }
  void ReshapeTensor(int model_id, int id, const string& csv_values) override;
  void AllocateTensors(int model_id) override;
  void ResetTensor(TfLiteTensor* tensor) override;
  void ResetTensor(int model_id, int id) override;
  void SetInput(int model_id, int id, const string& csv_values) override;
  void SetExpectation(int model_id, int id, const string& csv_values) override;
  void SetShapeExpectation(int model_id, int id, const string& csv_values) override;
  void Invoke(int model_id) override;
  void InvokeThroughPlanner(int model_id) override;
  bool CheckResults(int model_id) override;
  string ReadOutput(TfLiteTensor* tensor) override;
  string ReadOutput(int model_id, int id) override;
  void InvokeWithInput(std::vector<Job>& requests, std::vector<Tensors>& inputs, std::vector<Tensors>& outputs) override;
  void SetDataToTensor(TfLiteTensor* tensor, const string& csv_values) override;
  UniqueTfLiteTensor AllocateInputTensor(int subgraph_id, int index) override;
  UniqueTfLiteTensor AllocateOutputTensor(int subgraph_id, int index) override;
  bool NeedProfile() override {
    return interpreter_->NeedProfile();
  }

  void SetThreshold(double relative_threshold, double absolute_threshold);
  void SetQuantizationErrorMultiplier(int quantization_error_multiplier);

 private:
  void DeallocateStringTensor(TfLiteTensor* t) {
    if (t) {
      free(t->data.raw);
      t->data.raw = nullptr;
    }
  }
  void AllocateStringTensor(int id, size_t num_bytes, TfLiteTensor* t) {
    t->data.raw = reinterpret_cast<char*>(malloc(num_bytes));
    t->bytes = num_bytes;
    tensors_to_deallocate_[id] = t;
  }

  void ResetLSTMStateTensors();

  class DataExpectation;
  class ShapeExpectation;

  std::unique_ptr<OpResolver> resolver_;
  std::unique_ptr<Interpreter> interpreter_;
  // (model_id, (tensor_id, DataExpectation))
  std::map<int, std::map<int, std::unique_ptr<DataExpectation>>> expected_output_;
  // (model_id, (tensor_id, ShapeExpectation))
  std::map<int, std::map<int, std::unique_ptr<ShapeExpectation>>> expected_output_shape_;
  bool must_allocate_tensors_ = true;
  std::map<int, TfLiteTensor*> tensors_to_deallocate_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DRIVER_H_
