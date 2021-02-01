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

class MultiModelBenchmark {
 public:
	explicit MultiModelBenchmark(std::vector<int> device_plan) {
    kill_worker_ = false;
		device_plan_ = device_plan;

		std::string plan2str = "";
		for (int i = 0; i < device_plan.size(); ++i) {
			plan2str += std::to_string(device_plan[i]);
		}
    //log_file_.open("/data/local/tmp/log/model_execution_log.csv", std::fstream::app);
    log_file_.open("/data/local/tmp/log/model_execution_log"+plan2str+".csv", std::fstream::app);
    log_file_ << "batch_id\tmodel_name\tmodel_id\tdevice_id\tenqueue_time\tinvoke_time\tend_time\n";

    num_models_ = device_plan.size();
  }

  ~MultiModelBenchmark() {
    log_file_.close();
  }
	TfLiteStatus Worker(BenchmarkTfLiteModel benchmark, std::string graph_name);
	void GenerateRequests(int id);
  void Work(int id);
	TfLiteStatus Initialize(std::string graphs, std::string profile_data, int argc, char** argv);
	TfLiteStatus ParseJsonFile(std::string json_fname);
	TfLiteStatus RunRequests(int period);
	
	struct RuntimeConfig {
		int period_ms;
	};

	struct ModelConfig {
		std::string model_fname;
		int batch_size;
		int device;
    int64_t avg_time = 0;
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

  int batch_id = 0;
  int job_id = 0;

 private:
  std::vector<std::unique_ptr<BenchmarkTfLiteModel>> benchmarks_;
  std::vector<std::thread> threads_;
  std::ofstream log_file_;
	std::vector<ModelConfig> model_configs_;
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
};

TfLiteStatus MultiModelBenchmark::ParseJsonFile(std::string json_fname) {
  std::ifstream config(json_fname, std::ifstream::binary);

  Json::Value root;
  config >> root;

  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Please validate the json config file.";
    return kTfLiteError;
  }

  if (root["models"] == Json::Value::null) {
    TFLITE_LOG(ERROR) << "Please check if arguments `models` "
                      << "are given in the config file.";
    return kTfLiteError;
  }

  for (int i = 0; i < root["models"].size(); ++i) {
    ModelConfig model;
    Json::Value model_json_value = root["models"][i];
    model.model_fname = model_json_value["graph"].asString();

    // Set `batch_size`.
    // If no `batch_size` is given, the default batch size will be set to 1.
    if (model_json_value["batch_size"] != Json::Value::null) {
      model.batch_size = model_json_value["batch_size"].asInt();
    } else {
      model.batch_size = 1;
    }

    // Set `device`.
    // Fixes to the device if specified in case of `FixedDevicePlanner`.
    if (model_json_value["device"] != Json::Value::null) {
      model.device = model_json_value["device"].asInt();
    } else {
      model.device = -1;
    }

    model_configs_.push_back(model);
  }

  if (model_configs_.size() == 0) {
    TFLITE_LOG(ERROR) << "Please specify the name of TF Lite model files "
                      << "in `models` argument.";
    return kTfLiteError;
  }

  return kTfLiteOk;
}

/*
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
*/

TfLiteStatus MultiModelBenchmark::Worker(BenchmarkTfLiteModel benchmark, std::string graph_name) {
  benchmark.params_.Set<std::string>("graph", graph_name);
  TF_LITE_ENSURE_STATUS(benchmark.PrepareRun());
}

void MultiModelBenchmark::GenerateRequests(int id) {
  std::lock_guard<std::mutex> lock(*worker_mtxs_[id]);
  Job job;
  job.batch_id = batch_id++;
  job.id = id;
  job.device = device_plan_[id];
  job.avg_time = model_configs_[id].avg_time;

	int arc_mbn_requests = model_configs_[1].batch_size;
	int arc_res_requests = model_configs_[2].batch_size;
	int icn_requests = model_configs_[3].batch_size;

	int detection_batch = model_configs_[0].batch_size;

	std::vector<int> mbn_batch_plan(detection_batch, 0);
	std::vector<int> res_batch_plan(detection_batch, 0);
	std::vector<int> icn_batch_plan(detection_batch, 0);

	for (int i = 0; i < arc_mbn_requests; ++i)
		mbn_batch_plan[std::rand() % detection_batch] += 1;
	for (int i = 0; i < arc_res_requests; ++i)
		res_batch_plan[std::rand() % detection_batch] += 1;
	for (int i = 0; i < icn_requests; ++i)
		icn_batch_plan[std::rand() % detection_batch] += 1;

  job.enqueue_time = profiling::time::NowMicros();
  for (int k = 0; k < model_configs_[id].batch_size; ++k) {
		int run_arc_mbn = mbn_batch_plan[k];
		int run_arc_res = res_batch_plan[k];
		int run_icn = icn_batch_plan[k];

		job.next_requests.clear();
		job.next_requests.push_back(run_arc_mbn);
		job.next_requests.push_back(run_arc_res);
		job.next_requests.push_back(run_icn);

    worker_requests_[id].push_back(job);
    std::lock_guard<std::mutex> lock(requests_mtx_);
    num_requests_++;
  }
  (*worker_cvs_[id]).notify_all();
}

void MultiModelBenchmark::Work(int id) {
  std::thread t([this, id]() {
    while (true) {
      std::unique_lock<std::mutex> lock(*worker_mtxs_[id]);
      (*worker_cvs_[id]).wait(lock, [this, id]() {
          return kill_worker_ || !worker_requests_[id].empty();
      });


      if (worker_requests_[id].empty()) {
        lock.unlock();
        return;
      }

      Job job = worker_requests_[id].front();
      worker_requests_[id].pop_front();
      lock.unlock();

      int device = job.device;
      int64_t to_sleep_us = job.avg_time;

      job.invoke_time = profiling::time::NowMicros();
      benchmarks_[id]->RunImpl();
      // std::this_thread::sleep_for(std::chrono::microseconds(to_sleep_us));
      job.end_time = profiling::time::NowMicros();

      // Extra Requests
      if (id == 0) {
				for (int j = 0; j < 3; ++j) {
					int model_id = j + 1;
        	std::lock_guard<std::mutex> lock(*worker_mtxs_[model_id]);
					for (int k = 0; k < job.next_requests[j]; ++k) {
            Job following_job;
            following_job.batch_id = job.batch_id;
            following_job.id = model_id;
            following_job.device = device_plan_[model_id];
            following_job.avg_time = model_configs_[model_id].avg_time;
            following_job.enqueue_time = profiling::time::NowMicros();

						worker_requests_[model_id].push_back(following_job);
						std::lock_guard<std::mutex> lock(requests_mtx_);
						num_requests_++;
					}
          (*worker_cvs_[model_id]).notify_all();
				}
      }
      if (id == 3) {
        int model_id = 2;
        int batch_size = model_configs_[model_id].batch_size;

        {
          std::lock_guard<std::mutex> lock(*worker_mtxs_[model_id]);
          Job following_job;
          following_job.batch_id = job.batch_id;
          following_job.id = model_id;
          following_job.device = device_plan_[model_id];
          following_job.avg_time = model_configs_[model_id].avg_time;
          following_job.enqueue_time = profiling::time::NowMicros();
          worker_requests_[model_id].push_back(following_job);
          {
            std::lock_guard<std::mutex> request_lock(requests_mtx_);
            num_requests_++;
          }
          (*worker_cvs_[model_id]).notify_all();
        }
      }

      {
        std::lock_guard<std::mutex> main_lock(main_mtx_);
        jobs_finished_.push_back(job);
        cnt_++;
        jobs_finished_cv_.notify_all();
      }

    }
  });

  threads_.push_back(std::move(t));
}

TfLiteStatus MultiModelBenchmark::RunRequests(int period) {
  int run_time = 30 * 1000;

	for (int i = 0; i < benchmarks_.size(); ++i) {
    Work(i);
  }

  int64_t start_time = profiling::time::NowMicros();

  while(true) {
    int64_t current_time = profiling::time::NowMicros();
    if (current_time - start_time > run_time * 1000)
      break;

  	for (int i = 0; i < benchmarks_.size(); ++i) {
      if (i != 0)
        continue;
      GenerateRequests(i);
    }
  
    std::this_thread::sleep_for(std::chrono::milliseconds(period));
  }

  {
    std::unique_lock<std::mutex> main_lock(main_mtx_);
    jobs_finished_cv_.wait(main_lock, [this]() {
        return num_requests_ <= cnt_;
    });
  }

  kill_worker_ = true;
	for (int i = 0; i < benchmarks_.size(); ++i) {
    (*worker_cvs_[i]).notify_all(); 
  }

  for (std::thread& t : threads_) {
    t.join();
  }

  for (int i = 0; i < jobs_finished_.size(); ++i) {
    Job& job = jobs_finished_[i];
    std::string model_name = model_configs_[job.id].model_fname;
    log_file_ << job.batch_id << "\t"
              << model_name << "\t" 
              << job.id << "\t" 
              << device_plan_[job.id] << "\t"
              << job.enqueue_time << "\t"
              << job.invoke_time << "\t"
              << job.end_time << "\n";
  }

  return kTfLiteOk;
}

TfLiteStatus MultiModelBenchmark::Initialize(std::string json_path, std::string profile_data, int argc, char** argv) {
  TF_LITE_ENSURE_STATUS(ParseJsonFile(json_path));

  std::cout << "Parse Config File " << std::endl;
  std::ifstream profile_config(profile_data, std::ifstream::binary);
  Json::Value profile_root;
  profile_config >> profile_root;

  std::cout << profile_root << std::endl;

  for (auto& model_config : model_configs_) {
		std::string graph_name = model_config.model_fname;
    benchmarks_.emplace_back(new BenchmarkTfLiteModel());
    int last_idx = benchmarks_.size() - 1;
    benchmarks_[last_idx]->ParseFlags(argc, argv);
    benchmarks_[last_idx]->params_.Set<std::string>("graph", graph_name);

    model_config.device = device_plan_[last_idx];
    model_config.avg_time = profile_root[model_config.model_fname][std::to_string(model_config.device)].asInt();


    // Device Placement
    if (device_plan_[last_idx] % NUM_DEVICES == 1) {
      benchmarks_[last_idx]->params_.Set<bool>("use_gpu", true);
    } else if (device_plan_[last_idx] % NUM_DEVICES == 2) {
      benchmarks_[last_idx]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[last_idx]->params_.Set<std::string>("nnapi_accelerator_name", "qti-dsp");
    } else if (device_plan_[last_idx] % NUM_DEVICES == 3) {
      benchmarks_[last_idx]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[last_idx]->params_.Set<std::string>("nnapi_accelerator_name", "google-edgetpu");
    }

    std::unique_ptr<std::condition_variable> cv_ptr;
    cv_ptr.reset(new std::condition_variable());
    worker_cvs_.emplace_back(std::move(cv_ptr));

    std::unique_ptr<std::mutex> mtx_ptr;
    mtx_ptr.reset(new std::mutex());
    worker_mtxs_.emplace_back(std::move(mtx_ptr));

    std::deque<Job> dq;
    worker_requests_.push_back(dq);

    TF_LITE_ENSURE_STATUS(benchmarks_[last_idx]->PrepareRun());
  }

	return kTfLiteOk;
}

int Main(int argc, char** argv) {
  TFLITE_LOG(INFO) << "STARTING!!";
  BenchmarkTfLiteModel parser;
  TF_LITE_ENSURE_STATUS(parser.ParseFlags(argc, argv));
  int period = parser.params_.Get<int>("period_ms");
  std::string json_path = parser.params_.Get<std::string>("json_path");
  std::string profile_data = parser.params_.Get<std::string>("profile_data");

  std::ifstream config(json_path, std::ifstream::binary);
  Json::Value root;
  config >> root;

  int num_models = root["models"].size();

  std::cout << "Read JSON Config " << std::endl;
  std::cout << root << std::endl;
  
  std::vector<int> range;
  for (int i = 0; i < num_models; ++i) {
    range.push_back(i);
  }

  int sleep_time = 30 * 1000;
  bool first = true;
  bool fail = false;
  std::vector<std::string> executed_plans;

	std::srand(5323);
	do {
    /*
    if (!first && !fail) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    } else {
      first = false;
    }*/

		std::vector<int> device_plan;
    std::string current_plan = "";
		for (auto it = range.begin(); it != range.end(); ++it) {
			device_plan.push_back(*it);

      if (*it == 0)
        current_plan += "0";
      if (*it == 1)
        current_plan += "1";
      if (*it == 2)
        current_plan += "2";
      if (*it == 3)
        current_plan += "3";
      /*
      if (*it == 0 || *it == 1)
        current_plan += "0";
      else
        current_plan += "1";
      */
    }

    if (current_plan == "0123") {
      MultiModelBenchmark multimodel_benchmark(device_plan);
      TfLiteStatus status = multimodel_benchmark.Initialize(json_path, profile_data, argc, argv);
      if (status == kTfLiteOk) {
        multimodel_benchmark.RunRequests(period);
        fail = false;
      }
      else {
        fail = true;
      }
    }
    /*
    if (std::find(executed_plans.begin(), executed_plans.end(), current_plan) == executed_plans.end()) {
      executed_plans.push_back(current_plan);

      MultiModelBenchmark multimodel_benchmark(device_plan);
      TfLiteStatus status = multimodel_benchmark.Initialize(json_path, profile_data, argc, argv);
      if (status == kTfLiteOk) {
        multimodel_benchmark.RunRequests(period);
        fail = false;
      }
      else {
        fail = true;
      }
    }*/
	} while (std::next_permutation(range.begin(), range.end()));

  return EXIT_SUCCESS;
}
}  // namespace benchmark
}  // namespace tflite

int main(int argc, char** argv) { return tflite::benchmark::Main(argc, argv); }
