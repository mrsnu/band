#include "band/latency_estimator.h"

#include "band/context.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/profiler.h"
#include "band/worker.h"

namespace Band {
LatencyEstimator::LatencyEstimator(Context* context) : context_(context) {}

BandStatus LatencyEstimator::Init(const ProfileConfig& config) {
  profile_data_path_ = config.profile_data_path;
  profile_database_json_ = LoadJsonObjectFromFile(config.profile_data_path);
  // we cannot convert the model name strings to integer ids yet,
  // (profile_database_json_ --> profile_database_)
  // since we don't have anything in model_configs_ at the moment

  // Set how many runs are required to get the profile results.
  profile_online_ = config.online;
  profile_num_warmups_ = config.num_warmups;
  profile_num_runs_ = config.num_runs;
  profile_copy_computation_ratio_ = config.copy_computation_ratio;

  return kBandOk;
}

void LatencyEstimator::UpdateLatency(const SubgraphKey& key, int64_t latency) {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    int64_t prev_latency = it->second.moving_averaged;
    profile_database_[key].moving_averaged =
        profile_smoothing_factor_ * latency +
        (1 - profile_smoothing_factor_) * prev_latency;
  }
}

BandStatus LatencyEstimator::ProfileModel(ModelId model_id) {
  if (profile_online_) {
    for (WorkerId worker_id = 0; worker_id < context_->GetNumWorkers();
         worker_id++) {
      Worker* worker = context_->GetWorker(worker_id);
      // pause worker for profiling, must resume before continue
      worker->Pause();
      // wait for workers to finish current job
      worker->Wait();

      // TODO(dostos): find largest subgraph after subgraph-support (L878-,
      // tensorflow_band/lite/interpreter.cc)
      Job largest_subgraph_job(model_id);
      SubgraphKey subgraph_key =
          context_->GetModelSubgraphKey(model_id, worker_id);
      largest_subgraph_job.subgraph_key = subgraph_key;

      Profiler average_profiler;
      // invoke target subgraph in an isolated thread
      std::thread profile_thread([&]() {
        if (SetCPUThreadAffinity(worker->GetWorkerThreadAffinity()) !=
            kBandOk) {
          BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                            "Failed to propagate thread affinity of worker id "
                            "%d to profile thread",
                            worker_id);
        }

        // TODO(BAND-20): propagate affinity to CPU backend if necessary
        // (L1143-,tensorflow_band/lite/interpreter.cc)

        for (int i = 0; i < profile_num_warmups_; i++) {
          if (context_->Invoke(subgraph_key) != kBandOk) {
            BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                              "Profiler failed to invoke largest subgraph of "
                              "model %d in worker %d",
                              model_id, worker_id);
            return;
          }
        }

        for (int i = 0; i < profile_num_runs_; i++) {
          const size_t event_id = average_profiler.BeginEvent();
          if (context_->Invoke(subgraph_key) != kBandOk) {
            BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                              "Profiler failed to invoke largest subgraph of "
                              "model %d in worker %d",
                              model_id, worker_id);
            return;
          }
          average_profiler.EndEvent(event_id);
        }
      });
      profile_thread.join();

      // TODO(dostos): estimate latency with largest subgraph latency (L926-,
      // tensorflow_band/lite/interpreter.cc)
      if (average_profiler.GetNumEvents() != profile_num_runs_) {
        return kBandError;
      }

      const int64_t latency =
          average_profiler.GetAverageElapsedTime<std::chrono::microseconds>();

      profile_database_[subgraph_key] = {latency, latency};

      // resume worker
      worker->Resume();

      BAND_LOG_INTERNAL(
          BAND_LOG_INFO,
          "Estimated Latency\n model=%d avg=%d us worker=%d device=%s "
          "start=%s end=%s.",
          model_id, latency, worker_id,
          BandDeviceGetName(worker->GetDeviceFlag()),
          subgraph_key.GetInputOpsString().c_str(),
          subgraph_key.GetOutputOpsString().c_str());
    }
  }
  return kBandOk;
}

int64_t LatencyEstimator::GetProfiled(const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second.profiled;
  } else {
    return -1;
  }
}

int64_t LatencyEstimator::GetExpected(const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second.moving_averaged;
  } else {
    return -1;
  }
}
}  // namespace Band