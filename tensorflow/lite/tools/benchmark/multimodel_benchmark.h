#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_MULTIMODEL_BENCHMARK_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_MULTIMODEL_BENCHMARK_H_

#include <thread>
#include <string>
#include <vector>
#include <fstream>
#include <json/json.h>
#include <condition_variable>
#include <mutex>
#include <deque>
#include <iostream>
#include <cstdlib>

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/profiling/time.h"

#define NUM_DEVICES 4

namespace tflite {
namespace benchmark {

struct ModelConfig {
  std::string model_fname;
  int batch_size;
  int device = -1;
  int64_t avg_time = 0;
};

struct RuntimeConfig {
  int num_models;
  int period_ms;
  int run_duration;
  std::string log_path;
  std::string model_profile;
  Json::Value profile_result;
  std::vector<ModelConfig> model_configs;
};

struct Job {
  int job_id;
  int batch_id;
  int id;
  int64_t enqueue_time;
  int64_t invoke_time;
  int64_t end_time;
  int device;
  int64_t avg_time = 0;

  std::vector<int> next_requests;
};

TfLiteStatus ParseJsonFile(std::string json_path, RuntimeConfig& runtime_config);

class MultimodelBenchmark {
 public:
	explicit MultimodelBenchmark(RuntimeConfig runtime_config, std::vector<int> device_plan) {
    runtime_config_ = runtime_config;
    kill_worker_ = false;
		device_plan_ = device_plan;

		std::string plan2str = "";
		for (int i = 0; i < device_plan.size(); ++i) {
			plan2str += std::to_string(device_plan[i]);
		}
    log_file_.open("/data/local/tmp/model_execution_log.csv", std::fstream::app);
    // log_file_.open("/data/local/tmp/log/model_execution_log"+plan2str+".csv", std::fstream::app);
    log_file_ << "batch_id\tmodel_name\tmodel_id\tdevice_id\tenqueue_time\tinvoke_time\tend_time\n";

    num_models_ = device_plan.size();
    std::srand(5323);
  }

  ~MultimodelBenchmark() {
    log_file_.close();
  }

	void GenerateRequests(int id);
  void Work(int id);
	TfLiteStatus Initialize(int argc, char** argv);
	TfLiteStatus RunRequests();
	void DumpExecutionData();
	
  int batch_id = 0;
  int job_id = 0;

 private:
  std::vector<std::unique_ptr<BenchmarkTfLiteModel>> benchmarks_;
  std::vector<std::thread> threads_;
  std::ofstream log_file_;
  std::mutex main_mtx_;
  std::vector<Job> jobs_finished_;
  std::vector<std::unique_ptr<std::condition_variable>> worker_cvs_;
  std::vector<std::unique_ptr<std::mutex>> worker_mtxs_;
  std::vector<std::deque<Job>> worker_requests_;
  bool kill_worker_;
	std::vector<int> device_plan_;
  int num_models_;

  std::condition_variable jobs_finished_cv_;
  std::mutex requests_mtx_;
  std::mutex cnt_mtx_;
  int num_requests_ = 0;
  int cnt_ = 0;

	RuntimeConfig runtime_config_;
};

}
}

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_MULTIMODEL_BENCHMARK_H_
