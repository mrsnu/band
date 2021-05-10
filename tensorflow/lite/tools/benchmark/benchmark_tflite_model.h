/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <json/json.h>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

namespace tflite {
namespace benchmark {

// Benchmarks a TFLite model by running tflite interpreter.
class BenchmarkTfLiteModel : public BenchmarkModel {
 public:
  struct InputLayerInfo {
    InputLayerInfo() : has_value_range(false) {}

    std::string name;
    std::vector<int> shape;

    // The input value is randomly generated when benchmarking the NN model.
    // However, the NN model might require the value be limited to a certain
    // range [low, high] for this particular input layer. For simplicity,
    // support integer value first.
    bool has_value_range;
    int low;
    int high;

    // The input value will be loaded from 'input_file_path' INSTEAD OF being
    // randomly generated. Note the input file will be opened in binary mode.
    std::string input_file_path;
  };

  explicit BenchmarkTfLiteModel(BenchmarkParams params = DefaultParams());
  ~BenchmarkTfLiteModel() override;

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  TfLiteStatus ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  TfLiteStatus Init() override;
  TfLiteStatus RunImpl() override;
  TfLiteStatus RunImpl(int i) override;
  TfLiteStatus RunAll() override;
  static BenchmarkParams DefaultParams();
  TfLiteStatus RunModelsSync(std::vector<Job> requests);
  TfLiteStatus RunModelsAsync(std::vector<Job> requests);
  void WaitAsync();

 protected:
  TfLiteStatus PrepareInputData() override;
  TfLiteStatus ResetInputsAndOutputs() override;

  int64_t MayGetModelFileSize() override;

  virtual TfLiteStatus LoadModel(std::string graph);

  // Allow subclasses to create a customized Op resolver during init.
  virtual std::unique_ptr<tflite::OpResolver> GetOpResolver() const;

  // Allow subclass to initialize a customized tflite interpereter.
  virtual TfLiteStatus InitInterpreter();

  // Create a BenchmarkListener that's specifically for TFLite profiling if
  // necessary.
  virtual std::unique_ptr<BenchmarkListener> MayCreateProfilingListener() const;

  void CleanUp();

  std::unique_ptr<tflite::FlatBufferModel> model_;

  // A map for tracking which model name corresponds to which integer id.
  std::map<std::string, int> model_name_to_id_;

  // Map structure to find FlatBufferModel pointer with a model file name.
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models_;

  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::ExternalCpuBackendContext> external_context_;

 private:
  // Implement type erasure with unique_ptr with custom deleter.
  using VoidUniquePtr = std::unique_ptr<void, void (*)(void*)>;

  struct InputTensorData {
    InputTensorData() : data(nullptr, nullptr) {}

    VoidUniquePtr data;
    size_t bytes;
  };

  template <typename T, typename Distribution>
  inline InputTensorData CreateInputTensorData(int num_elements,
                                               Distribution distribution) {
    InputTensorData tmp;
    tmp.bytes = sizeof(T) * num_elements;
    T* raw = new T[num_elements];
    std::generate_n(raw, num_elements, [&]() {
      return static_cast<T>(distribution(random_engine_));
    });
    tmp.data = VoidUniquePtr(static_cast<void*>(raw),
                             [](void* ptr) { delete[] static_cast<T*>(ptr); });
    return tmp;
  }

  InputTensorData CreateRandomTensorData(const TfLiteTensor& t,
                                         const InputLayerInfo* layer_info);

  InputTensorData LoadInputTensorData(const TfLiteTensor& t,
                                      const std::string& input_file_path);

  TfLiteStatus ParseJsonFile();

  // Convert model name strings to integer ids for the given model profiles.
  // The return val can be given to the interpreter via Interpreter::Profile().
  Interpreter::ModelDeviceToLatency ConvertModelNameToId(const Json::Value name_profile);

  // Convert model integer ids back to string-type names for model profiles.
  // This function does not erase entries in name_profile for models that were
  // not run during this benchmark run.
  void ConvertModelIdToName(const Interpreter::ModelDeviceToLatency id_profile,
                            Json::Value& name_profile);

  // spawn a thread that generates input requests periodically for all models
  void GeneratePeriodicRequests();

  std::vector<InputLayerInfo> inputs_;
  std::vector<InputTensorData> inputs_data_;
  std::unique_ptr<BenchmarkListener> profiling_listener_ = nullptr;
  std::unique_ptr<BenchmarkListener> ruy_profiling_listener_ = nullptr;
  std::mt19937 random_engine_;
  std::vector<Interpreter::TfLiteDelegatePtr> owned_delegates_;
  // Always TFLITE_LOG the benchmark result.
  BenchmarkLoggingListener log_output_;

  // boolean flag for letting child threads know that it's time to go home
  bool kill_app_ = false;
};

class LoadGenImpl : public util::LoadGen {
 public:
  explicit LoadGenImpl(BenchmarkTfLiteModel* benchmark_model)
    : benchmark_model_(benchmark_model) {}
  TfLiteStatus RunModelsSync(std::vector<Job> requests) {
    return benchmark_model_->RunModelsSync(requests);
  }

  TfLiteStatus RunModelsAsync(std::vector<Job> requests) {
    return benchmark_model_->RunModelsAsync(requests);
  }

  void Wait() {
    return benchmark_model_->WaitAsync();
  }

 private:
  BenchmarkTfLiteModel* benchmark_model_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
