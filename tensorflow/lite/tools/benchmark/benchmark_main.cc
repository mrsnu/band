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

#include <thread>
#include <string>
#include <vector>
#include <fstream>
#include <condition_variable>
#include <mutex>
#include <deque>

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/profiling/time.h"

#define NUM_DEVICES 4
namespace tflite {
namespace benchmark {


struct Job {
  int model_id;
  int64_t enqueue_time;
  int64_t invoke_time;
  int64_t end_time;
  int device;

  explicit Job(int model_id, int device) : model_id(model_id), device(device) {
    enqueue_time = profiling::time::NowMicros();
  }
};

// The job queue which can be shared by multiple threads.
struct ConcurrentJobQueue {
  std::deque<Job> queue;
  std::mutex mtx;
  std::condition_variable cv;
  bool kill = false;
};

class MultiModelBenchmark {
 public:
	explicit MultiModelBenchmark(std::string log_path) {
    log_file_.open(log_path);
    log_file_ << "sched_id\t"
         << "model_name\t"
         << "model_id\t"
         << "device_id\t"
         << "start_idx\t"
         << "end_idx\t"
         << "subgraph_idx\t"
         << "enqueue_time\t"
         << "invoke_time\t"
         << "end_time\t"
         << "profiled_time\t"
         << "expected_latency\t"
         << "slo_us\t"
         << "job status\t"
         << "is_final_subgraph\n";
  }
  ~MultiModelBenchmark() {
    log_file_.close();
  }
	TfLiteStatus Worker(BenchmarkTfLiteModel benchmark, std::string graph_name);
	void GenerateRequests(int id, std::string graph_name);
	TfLiteStatus Initialize(std::string graphs, int device, int argc, char** argv);
	TfLiteStatus ParseGraphFileNames(std::string graphs);
	TfLiteStatus RunRequests(int period, int duration_ms, int start_at);
	void RunStream(int duration);

 private:
	std::vector<std::string> graph_names_;
  std::vector<std::unique_ptr<BenchmarkTfLiteModel>> benchmarks_;
  std::vector<std::thread> threads_;
  std::vector<std::unique_ptr<ConcurrentJobQueue>> queues_;
  std::ofstream log_file_;
};

TfLiteStatus MultiModelBenchmark::ParseGraphFileNames(std::string graphs) {
  size_t previous = 0, current;

  do {
    current = graphs.find(',', previous);
    std::string graph = graphs.substr(previous, current - previous);
    if (graph.size() > 0)
      graph_names_.push_back(graph);
    previous = current + 1;
  } while (current != string::npos);

  if (graph_names_.size() == 0) {
    TFLITE_LOG(ERROR) << "Please specify the name of TF Lite input files.";
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus MultiModelBenchmark::Worker(BenchmarkTfLiteModel benchmark, std::string graph_name) {
  benchmark.params_.Set<std::string>("graph", graph_name);
  TF_LITE_ENSURE_STATUS(benchmark.PrepareRun());
}

void MultiModelBenchmark::GenerateRequests(int id, std::string graph_name) {
  std::thread t([this, id, graph_name]() {
    int sched_id = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(queues_[id]->mtx);
      queues_[id]->cv.wait(lock, [this, id] {
        return !queues_[id]->queue.empty() || queues_[id]->kill;
      });
      if (queues_[id]->queue.empty()) {
        return;
      }

      Job job = queues_[id]->queue.front();
      queues_[id]->queue.pop_front();
      lock.unlock();

      job.invoke_time = profiling::time::NowMicros();
      benchmarks_[id]->RunImpl();
      job.end_time = profiling::time::NowMicros();
 
      // Logging
      lock.lock();
      log_file_ << sched_id++ << "\t"
                << graph_name << "\t"
                << id << "\t"
                << job.device << "\t"
                << -1 << "\t"
                << -1 << "\t"
                << -1 << "\t"
                << job.enqueue_time << "\t"
                << job.invoke_time << "\t"
                << job.end_time << "\t"
                << -1 << "\t"
                << -1 << "\t"
                << -1 << "\t"
                << -1 << "\t"
                << -1 << "\n";     
      lock.unlock();
    };
  });
  threads_.push_back(std::move(t));
}

TfLiteStatus MultiModelBenchmark::RunRequests(int period, int duration_ms, int start_at) {
	for (int i = 0; i < benchmarks_.size(); ++i) {
    queues_.emplace_back(std::make_unique<ConcurrentJobQueue>());
		std::string graph_name = benchmarks_[i]->params_.Get<std::string>("graph");
    GenerateRequests(i, graph_name);
  }

  if (start_at > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(start_at * 1000));
  }
  int64_t start_time = profiling::time::NowMicros();
  int curr_time_ms = start_time / 1000;
  do {
    int64_t start_run = profiling::time::NowMicros();
    // Invoke Async
    for (int id = 0; id < benchmarks_.size(); ++id) {
      int device = benchmarks_[id]->params_.Get<int32_t>("device");
      std::unique_lock<std::mutex> lock(queues_[id]->mtx);
      queues_[id]->queue.push_back(Job(id, device));
      queues_[id]->cv.notify_all();
      lock.unlock();
    }

    int real_period = std::rand() % period + (period / 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(real_period));
    int64_t end_run = profiling::time::NowMicros();
    curr_time_ms = (end_run - start_time) / 1000;
  } while(curr_time_ms < duration_ms);

  for (int id = 0; id < benchmarks_.size(); ++id) {
    std::unique_lock<std::mutex> lock(queues_[id]->mtx);
    queues_[id]->kill = true;
    queues_[id]->cv.notify_all();
    lock.unlock();
  }

  for (std::thread& t : threads_) {
    t.join();
  }

  return kTfLiteOk;
}

void MultiModelBenchmark::RunStream(int duration_ms) {
  int id = 0;
  int sched_id = 0;
  int64_t start_time = profiling::time::NowMicros();
  int curr_time_ms = start_time / 1000;
  do {
    int64_t start_run = profiling::time::NowMicros();
    benchmarks_[id]->RunImpl();
    int64_t end_run = profiling::time::NowMicros();
    curr_time_ms = (end_run - start_time) / 1000;

    log_file_ << sched_id++ << "\t"
              << benchmarks_[id]->params_.Get<std::string>("graph") << "\t"
              << id << "\t"
              << benchmarks_[id]->params_.Get<int32_t>("device") << "\t"
              << -1 << "\t"
              << -1 << "\t"
              << -1 << "\t"
              << start_run << "\t"
              << start_run << "\t"
              << end_run << "\t"
              << -1 << "\t"
              << -1 << "\t"
              << -1 << "\t"
              << -1 << "\t"
              << -1 << "\n";
  } while(curr_time_ms < duration_ms);
}

TfLiteStatus MultiModelBenchmark::Initialize(std::string graphs, int device, int argc, char** argv) {
  TF_LITE_ENSURE_STATUS(ParseGraphFileNames(graphs));

  for (auto graph_name : graph_names_) {
    benchmarks_.emplace_back(new BenchmarkTfLiteModel());
    int last_idx = benchmarks_.size() - 1;
    benchmarks_[last_idx]->ParseFlags(argc, argv);
    benchmarks_[last_idx]->params_.Set<std::string>("graph", graph_name);
    benchmarks_[last_idx]->params_.Set<int32_t>("device", device);

    if (device == 1) {
      benchmarks_[last_idx]->params_.Set<bool>("use_gpu", true);
    } else if (device == 2) {
      benchmarks_[last_idx]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[last_idx]->params_.Set<std::string>("nnapi_accelerator_name", "qti-dsp");
    } else if (device == 3) {
      benchmarks_[last_idx]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[last_idx]->params_.Set<std::string>("nnapi_accelerator_name", "google-edgetpu");
    }

    TF_LITE_ENSURE_STATUS(benchmarks_[last_idx]->PrepareRun());
  }

	return kTfLiteOk;
}

int Main(int argc, char** argv) {
  TFLITE_LOG(INFO) << "STARTING!";
  BenchmarkTfLiteModel parser;
  TF_LITE_ENSURE_STATUS(parser.ParseFlags(argc, argv));
  // Currently, multi-model execution is not fully supported.
  std::string graphs = parser.params_.Get<std::string>("graphs");
  int device = parser.params_.Get<int>("device");

	MultiModelBenchmark multimodel_benchmark(
    parser.params_.Get<std::string>("log_path")
  );
	multimodel_benchmark.Initialize(graphs, device, argc, argv);


  std::string execution_mode = parser.params_.Get<std::string>("execution_mode");
  int duration = parser.params_.Get<int>("duration_ms");
  int start_at = parser.params_.Get<int>("start_at");
  if (execution_mode == "stream") {
    // Only single model execution is supported with the stream mode.
    multimodel_benchmark.RunStream(duration);
  } else if (execution_mode == "periodic") {
    int period = parser.params_.Get<int>("period");
    multimodel_benchmark.RunRequests(period, duration, start_at);
  } else {
    TFLITE_LOG(ERROR) << "Wrong execution mode.";
    return -1;
  }


  return EXIT_SUCCESS;
}
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) { return tflite::benchmark::Main(argc, argv); }
