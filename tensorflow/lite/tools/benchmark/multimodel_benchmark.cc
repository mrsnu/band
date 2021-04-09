#include "tensorflow/lite/tools/benchmark/multimodel_benchmark.h"

namespace tflite {
namespace benchmark {

TfLiteStatus ParseJsonFile(std::string json_path, RuntimeConfig& runtime_config) {
  std::ifstream config(json_path, std::ifstream::binary);
  Json::Value root;
  config >> root;
  std::cout << "Read JSON Config " << std::endl;
  
  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Please validate the json config file.";
    return kTfLiteError;
  }

  // Set Runtime Configurations
  // Optional
  if (!root["running_time_ms"].isNull())
    runtime_config.run_duration = root["running_time_ms"].asInt();
  if (!root["model_profile"].isNull()) {
    runtime_config.model_profile = root["model_profile"].asString();
    std::ifstream profile_config(runtime_config.model_profile, std::ifstream::binary);
    profile_config >> runtime_config.profile_result;
  }

  // Required
  if (root["period_ms"].isNull() ||
      root["log_path"].isNull() ||
      root["models"].isNull()) {
    TFLITE_LOG(ERROR) << "Please check if arguments "
                      << "`period_ms`, `log_path` and `models`"
                      << " are given in the config file.";
    return kTfLiteError;
  }

  runtime_config.period_ms = root["period_ms"].asInt();
  runtime_config.log_path = root["log_path"].asString();

  // Set Model Configurations
  for (int i = 0; i < root["models"].size(); ++i) {
    ModelConfig model;
    Json::Value model_json_value = root["models"][i];
    if (model_json_value["graph"].isNull()) {
      TFLITE_LOG(ERROR) << "Please check if argument `graph` is not given in "
                           "the model configs.";
      return kTfLiteError;
    }
    model.model_fname = model_json_value["graph"].asString();

    // Set `batch_size`.
    // If no `batch_size` is given, the default batch size will be set to 1.
    if (!model_json_value["batch_size"].isNull())
      model.batch_size = model_json_value["batch_size"].asInt();

    // Set `device`.
    if (!model_json_value["device"].isNull())
      model.device = model_json_value["device"].asInt();

    runtime_config.model_configs.push_back(model);
  }

  if (runtime_config.model_configs.size() == 0) {
    TFLITE_LOG(ERROR) << "Please specify at list one model "
                      << "in `models` argument.";
    return kTfLiteError;
  }
  runtime_config.num_models = runtime_config.model_configs.size();

  TFLITE_LOG(INFO) << root;

  return kTfLiteOk;
}

void MultimodelBenchmark::GenerateRequests(int id) {
  std::lock_guard<std::mutex> lock(*worker_mtxs_[id]);
  Job job;
  job.batch_id = batch_id++;
  job.id = id;
  job.device = device_plan_[id];
  job.avg_time = runtime_config_.model_configs[id].avg_time;

	int arc_mbn_requests = runtime_config_.model_configs[1].batch_size;
	int arc_res_requests = runtime_config_.model_configs[2].batch_size;
	int icn_requests = runtime_config_.model_configs[3].batch_size;

	int detection_batch = runtime_config_.model_configs[0].batch_size;

	std::vector<int> mbn_batch_plan(detection_batch, 0);
	std::vector<int> res_batch_plan(detection_batch, 0);
	std::vector<int> icn_batch_plan(detection_batch, 0);

	for (int i = 0; i < arc_mbn_requests; ++i) {
		mbn_batch_plan[std::rand() % detection_batch] += 1;
  }
	for (int i = 0; i < arc_res_requests; ++i) {
		res_batch_plan[std::rand() % detection_batch] += 1;
  }
	for (int i = 0; i < icn_requests; ++i) {
		icn_batch_plan[std::rand() % detection_batch] += 1;
  }

  job.enqueue_time = profiling::time::NowMicros();
  for (int k = 0; k < runtime_config_.model_configs[id].batch_size; ++k) {
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

void MultimodelBenchmark::Work(int id) {
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
            following_job.avg_time = runtime_config_.model_configs[model_id].avg_time;
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
        int batch_size = runtime_config_.model_configs[model_id].batch_size;

        {
          std::lock_guard<std::mutex> lock(*worker_mtxs_[model_id]);
          Job following_job;
          following_job.batch_id = job.batch_id;
          following_job.id = model_id;
          following_job.device = device_plan_[model_id];
          following_job.avg_time = runtime_config_.model_configs[model_id].avg_time;
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

TfLiteStatus MultimodelBenchmark::RunRequests() {
  int64_t start_time = profiling::time::NowMicros();
  int period = runtime_config_.period_ms;

  while(true) {
    int64_t current_time = profiling::time::NowMicros();
    if (current_time - start_time > runtime_config_.run_duration * 1000)
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

  DumpExecutionData();

  return kTfLiteOk;
}

void MultimodelBenchmark::DumpExecutionData() {
  for (int i = 0; i < jobs_finished_.size(); ++i) {
    Job& job = jobs_finished_[i];
    std::string model_name = runtime_config_.model_configs[job.id].model_fname;
    log_file_ << job.batch_id << "\t"
              << model_name << "\t" 
              << job.id << "\t" 
              << device_plan_[job.id] << "\t"
              << job.enqueue_time << "\t"
              << job.invoke_time << "\t"
              << job.end_time << "\n";
  }  
}

TfLiteStatus MultimodelBenchmark::Initialize(int argc, char** argv) {
  for (auto& model_config : runtime_config_.model_configs) {
		std::string graph_name = model_config.model_fname;
    benchmarks_.emplace_back(new BenchmarkTfLiteModel());
    int index = benchmarks_.size() - 1;
    benchmarks_[index]->ParseFlags(argc, argv);
    benchmarks_[index]->params_.Set<std::string>("graph", graph_name);

    if (runtime_config_.model_profile != "") {
      std::string model_fname = model_config.model_fname;
      std::string device_id = std::to_string(model_config.device);
      model_config.avg_time = runtime_config_.profile_result[model_fname][device_id].asInt(); 
    }

    if (model_config.device >= 0) {
      model_config.device = device_plan_[index];
    }
    if (model_config.device % NUM_DEVICES == 1) {
      // NOTE: GPUDelegate Settings in benchmark may differ from `GPUDelegateOptionsDefault`.
      benchmarks_[index]->params_.Set<bool>("use_gpu", true);
    } else if (model_config.device % NUM_DEVICES == 2) {
      benchmarks_[index]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[index]->params_.Set<std::string>("nnapi_accelerator_name", "qti-dsp");
    } else if (model_config.device % NUM_DEVICES == 3) {
      benchmarks_[index]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[index]->params_.Set<std::string>("nnapi_accelerator_name", "google-edgetpu");
    }

    std::unique_ptr<std::condition_variable> cv_ptr;
    cv_ptr.reset(new std::condition_variable());
    worker_cvs_.emplace_back(std::move(cv_ptr));

    std::unique_ptr<std::mutex> mtx_ptr;
    mtx_ptr.reset(new std::mutex());
    worker_mtxs_.emplace_back(std::move(mtx_ptr));

    std::deque<Job> dq;
    worker_requests_.push_back(dq);

    TF_LITE_ENSURE_STATUS(benchmarks_[index]->PrepareRun());

    for (int i = 0; i < benchmarks_.size(); ++i) {
      Work(i);
    }
  }

	return kTfLiteOk;
}

}
}
