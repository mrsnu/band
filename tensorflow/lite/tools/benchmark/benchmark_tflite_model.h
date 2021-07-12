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
  explicit BenchmarkTfLiteModel(BenchmarkParams params = DefaultParams());
  ~BenchmarkTfLiteModel() override;

  std::vector<Flag> GetFlags() override;
  void LogParams() override;
  TfLiteStatus ValidateParams() override;
  uint64_t ComputeInputBytes() override;
  TfLiteStatus Init() override;
  TfLiteStatus RunImpl(int i) override;
  TfLiteStatus RunAll() override;
  TfLiteStatus RunPeriodic() override;
  TfLiteStatus RunPeriodicSingleThread() override;
  TfLiteStatus RunStream() override;
  static BenchmarkParams DefaultParams();

 protected:
  TfLiteStatus PrepareInputData() override;

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

  // Map structure to find FlatBufferModel pointer with a model file name.
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models_;
  std::vector<std::vector<TfLiteTensor*>> model_input_tensors_;
  std::vector<std::vector<TfLiteTensor*>> model_output_tensors_;

  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::ExternalCpuBackendContext> external_context_;

 private:
  template <typename T, typename Distribution>
  inline util::InputTensorData CreateInputTensorData(int num_elements,
                                               Distribution distribution) {
    util::InputTensorData tmp;
    tmp.bytes = sizeof(T) * num_elements;
    T* raw = new T[num_elements];
    std::generate_n(raw, num_elements, [&]() {
      return static_cast<T>(distribution(random_engine_));
    });
    tmp.data = util::VoidUniquePtr(static_cast<void*>(raw),
                             [](void* ptr) { delete[] static_cast<T*>(ptr); });
    return tmp;
  }

  util::InputTensorData CreateRandomTensorData(const TfLiteTensor& t,
                                         const util::InputLayerInfo* layer_info);

  util::InputTensorData LoadInputTensorData(const TfLiteTensor& t,
                                      const std::string& input_file_path);

  // spawn threads that generate input requests periodically for all models
  void GeneratePeriodicRequests();

  // spawn a thread that generates input requests periodically for all models
  void GeneratePeriodicRequestsSingleThread();

  std::unique_ptr<BenchmarkListener> profiling_listener_ = nullptr;
  std::unique_ptr<BenchmarkListener> ruy_profiling_listener_ = nullptr;
  std::mt19937 random_engine_;
  std::vector<Interpreter::TfLiteDelegatePtr> owned_delegates_;
  // Always TFLITE_LOG the benchmark result.
  BenchmarkLoggingListener log_output_;

  // boolean flag for letting child threads know that it's time to go home
  bool kill_app_ = false;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_TFLITE_MODEL_H_
