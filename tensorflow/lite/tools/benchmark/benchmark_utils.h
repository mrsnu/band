/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_

#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <thread>

#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace benchmark {
namespace util {

// A convenient function that wraps tflite::profiling::time::SleepForMicros and
// simply return if 'sleep_seconds' is negative.
void SleepForSeconds(double sleep_seconds);

// Split the 'str' according to 'delim', and store each splitted element into
// 'values'.
template <typename T>
bool SplitAndParse(const std::string& str, char delim, std::vector<T>* values) {
  std::istringstream input(str);
  for (std::string line; std::getline(input, line, delim);) {
    std::istringstream to_parse(line);
    T val;
    to_parse >> val;
    if (!to_parse.eof() && !to_parse.good()) {
      return false;
    }
    values->emplace_back(val);
  }
  return true;
}

// Keeps the runtime configuration from json config file.
struct RuntimeConfig {
  RuntimeConfig() {
    // To avoid kTfLiteNumDevices dependent initialization
    for (int i = 0; i < kTfLiteNumDevices; i++) {
      worker_cpu_masks[i] = tflite::impl::kTfLiteNumCpuMasks;
    }
  }
  // Required
  std::string log_path;
  TfLitePlannerType planner_type;
  std::string execution_mode;
  // Optional
  impl::TfLiteCPUMaskFlags cpu_masks = impl::kTfLiteAll;
  impl::TfLiteCPUMaskFlags worker_cpu_masks[kTfLiteNumDevices];
  int running_time_ms = 60000;
  float profile_smoothing_factor = 0.1;
  std::string model_profile;
  bool allow_work_steal = false;
  int schedule_window_size = INT_MAX;
  std::vector<ModelConfig> model_configs;
};

class LoadGen {
 public:
  // Enqueue the `requests` and wait until the execution is finished.
  virtual TfLiteStatus RunModelsSync(std::vector<Job> requests) = 0;
  // Enqueue the `requests`.
  virtual TfLiteStatus RunModelsAsync(std::vector<Job> requests) = 0;
  // Wait.
  virtual void Wait() = 0;
  // Parse user given json config file.
  TfLiteStatus ParseJsonFile(std::string json_fname);
  RuntimeConfig GetRuntimeConfig() {
    return runtime_config_;
  }
  std::vector<Job> GetRequests() {
    std::vector<Job> requests;
    for (auto& model_config : runtime_config_.model_configs) {
      int model_id = model_config.model_id;
      for (int k = 0; k < model_config.batch_size; ++k) {
        requests.push_back(Job(model_id));
      }
    }
    return requests;
  }
  // Run requests in back-to-back manner for `running_time_ms` milli-seconds.
  // NOTE the workload is static.
  TfLiteStatus RunStream() {
    int run_duration_us = runtime_config_.running_time_ms * 1000;
    int num_frames = 0;
    int64_t start = profiling::time::NowMicros();
    while (true) {
      TF_LITE_ENSURE_STATUS(RunModelsSync(GetRequests()));
      int64_t current = profiling::time::NowMicros();
      num_frames++;
      if (current - start >= run_duration_us)
        break;
    }
    int64_t end = profiling::time::NowMicros();
    float time_taken = static_cast<float>(end - start);
    TFLITE_LOG(INFO) << "# processed frames: " << num_frames;
    TFLITE_LOG(INFO) << "Time taken (us): " << time_taken;
    TFLITE_LOG(INFO) << "Measured FPS: "
                     << (num_frames / time_taken) * 1000000;

    return kTfLiteOk;
  }

  TfLiteStatus RunPeriodic() {
    // initialize values in case this isn't our first run
    kill_app_ = false;

    // spawn a child thread to do our work, since we're going to sleep
    // Note: spawning a separate thread is technically unnecessary if we only
    // have a single thread that generate requests, but we may have multiple
    // threads doing that in the future so we might as well make the code easily
    // adaptable to such situtations.
    GeneratePeriodicRequests();

    // wait for 60 seconds until we stop the benchmark
    // we could set a command line arg for this value as well
    std::this_thread::sleep_for(
        std::chrono::milliseconds(runtime_config_.running_time_ms));
    kill_app_ = true;
    Wait();
    return kTfLiteOk;
  }

 private:
  void GeneratePeriodicRequests() {
    for (auto& model_config : runtime_config_.model_configs) {
      int model_id = model_config.model_id,
          batch_size = model_config.batch_size,
          period_ms = model_config.period_ms;

      std::thread t([this, batch_size, model_id, period_ms]() {
        std::vector<Job> requests(batch_size, Job(model_id));
        while (true) {
          // measure the time it took to generate requests
          int64_t start = profiling::time::NowMicros();
          RunModelsAsync(requests);
          // interpreter_->InvokeModelsAsync(requests);
          int64_t end = profiling::time::NowMicros();
          int duration_ms = (end - start) / 1000;

          // sleep until we reach period_ms
          if (duration_ms < period_ms) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(period_ms - duration_ms));
          }

          if (kill_app_) return;
        }
      });

      t.detach();
    }
  }

  RuntimeConfig runtime_config_;
  bool kill_app_ = false;
};

}  // namespace util
}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_UTILS_H_
