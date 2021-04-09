#include "tensorflow/lite/tools/benchmark/multimodel_benchmark.h"

namespace tflite {
namespace benchmark {
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

TfLiteStatus MultimodelBenchmark::RunRequests(int period) {
	for (int i = 0; i < benchmarks_.size(); ++i) {
    Work(i);
  }

  int64_t start_time = profiling::time::NowMicros();

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

  return kTfLiteOk;
}

TfLiteStatus MultimodelBenchmark::Initialize(int argc, char** argv) {
  for (auto& model_config : runtime_config_.model_configs) {
		std::string graph_name = model_config.model_fname;
    benchmarks_.emplace_back(new BenchmarkTfLiteModel());
    int index = benchmarks_.size() - 1;
    benchmarks_[index]->ParseFlags(argc, argv);
    benchmarks_[index]->params_.Set<std::string>("graph", graph_name);

    model_config.device = device_plan_[index];
    if (runtime_config_.model_profile != "") {
      std::cout << "PROFILE RESULTS" << std::endl;
      model_config.avg_time = runtime_config_.profile_result[model_config.model_fname][std::to_string(model_config.device)].asInt(); 
    }

    // Device Placement
    if (device_plan_[index] % NUM_DEVICES == 1) {
      benchmarks_[index]->params_.Set<bool>("use_gpu", true);
    } else if (device_plan_[index] % NUM_DEVICES == 2) {
      benchmarks_[index]->params_.Set<bool>("use_nnapi", true);
      benchmarks_[index]->params_.Set<std::string>("nnapi_accelerator_name", "qti-dsp");
    } else if (device_plan_[index] % NUM_DEVICES == 3) {
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
  }

	return kTfLiteOk;
}

}
}
