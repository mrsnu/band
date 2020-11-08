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

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace benchmark {

class MultiModelBenchmark {
 public:
	explicit MultiModelBenchmark() {};
	TfLiteStatus Worker(BenchmarkTfLiteModel benchmark, std::string graph_name);
	void GenerateRequests(int id, int interval, std::string graph_name, int run_time);
	TfLiteStatus Initialize(std::string graphs, int argc, char** argv);
	TfLiteStatus ParseGraphFileNames(std::string graphs);
	TfLiteStatus RunRequests(int period);

 private:
	std::vector<std::string> graph_names_;
  std::vector<std::unique_ptr<BenchmarkTfLiteModel>> benchmarks_;
  std::vector<std::thread> threads_;
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

void MultiModelBenchmark::GenerateRequests(int id, int interval, std::string graph_name, int run_time) {
  std::thread t([this, id, interval, graph_name, run_time]() {
    int iterations = run_time / interval;
    for (int i = 0; i < iterations; ++i) {
      int64_t exe_time = benchmarks_[id]->RunIteration();
      int duration = exe_time / 1000;
      if (duration < interval) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval - duration));
      }
    }
  });
  threads_.push_back(std::move(t));
}

TfLiteStatus MultiModelBenchmark::RunRequests(int period) {
  int run_time = 6000;

	for (int i = 0; i < benchmarks_.size(); ++i){
		std::string graph_name = benchmarks_[i]->params_.Get<std::string>("graph");
    GenerateRequests(i, period, graph_name, run_time);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(run_time));
  for (std::thread& t : threads_) {
    t.join();
  }

  return kTfLiteOk;
}

TfLiteStatus MultiModelBenchmark::Initialize(std::string graphs, int argc, char** argv) {
  TF_LITE_ENSURE_STATUS(ParseGraphFileNames(graphs));

  for (auto graph_name : graph_names_) {
    benchmarks_.emplace_back(new BenchmarkTfLiteModel());
    int last_idx = benchmarks_.size() - 1;
    benchmarks_[last_idx]->ParseFlags(argc, argv);
    benchmarks_[last_idx]->params_.Set<std::string>("graph", graph_name);
    TF_LITE_ENSURE_STATUS(benchmarks_[last_idx]->PrepareRun());
  }

	return kTfLiteOk;
}

int Main(int argc, char** argv) {
  TFLITE_LOG(INFO) << "STARTING!";
  BenchmarkTfLiteModel parser;
  TF_LITE_ENSURE_STATUS(parser.ParseFlags(argc, argv));
  int period = parser.params_.Get<int>("period");
  std::string graphs = parser.params_.Get<std::string>("graphs");

	MultiModelBenchmark multimodel_benchmark;
	multimodel_benchmark.Initialize(graphs, argc, argv);
	multimodel_benchmark.RunRequests(period);

  return EXIT_SUCCESS;
}
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) { return tflite::benchmark::Main(argc, argv); }
