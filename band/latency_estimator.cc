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
      SubgraphKey subgraph_key(model_id, worker_id);
      largest_subgraph_job.subgraph_key = subgraph_key;

      Profiler average_profiler;
      // invoke target subgraph in an isolated thread
      std::thread profile_thread([&]() {
        if (worker->GetWorkerThreadAffinity().NumEnabled() > 0 &&
            SetCPUThreadAffinity(worker->GetWorkerThreadAffinity()) !=
                kBandOk) {
          BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                            "Failed to propagate thread affinity of worker id "
                            "%d to profile thread",
                            worker_id);
          return;
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
  } else {
    const std::string model_name = context_->GetModelSpec(model_id)->path;
    auto model_profile = JsonToModelProfile(model_name, model_id);
    if (model_profile.size() > 0) {
      profile_database_.insert(model_profile.begin(), model_profile.end());
      BAND_LOG_INTERNAL(
          BAND_LOG_INFO,
          "Successfully found %d profile entries for model (%s, %d).",
          model_profile.size(), model_name.c_str(), model_id);
    } else {
      BAND_LOG_INTERNAL(
          BAND_LOG_WARNING,
          "Failed to find profile entries for given model name %s.",
          model_name.c_str());
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

BandStatus LatencyEstimator::DumpProfile() {
  return WriteJsonObjectToFile(ProfileToJson(), profile_data_path_);
}

size_t LatencyEstimator::GetProfileHash() const {
  auto hash_func = std::hash<int>();
  std::size_t hash = hash_func(context_->GetNumWorkers());
  for (int i = 0; i < context_->GetNumWorkers(); i++) {
    hash ^= hash_func(context_->GetWorker(i)->GetDeviceFlag());
    hash ^= hash_func(context_->GetWorker(i)->GetNumThreads());
    hash ^= hash_func(
        context_->GetWorker(i)->GetWorkerThreadAffinity().GetCPUMaskFlag());
  }
  return hash;
}

std::map<SubgraphKey, LatencyEstimator::Latency>
LatencyEstimator::JsonToModelProfile(const std::string& model_fname,
                                     const int model_id) {
  auto string_to_node_indices = [](std::string index_string) {
    std::set<int> node_indices;
    std::stringstream ss(index_string);

    for (int i; ss >> i;) {
      node_indices.insert(i);
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }

    return node_indices;
  };

  std::map<SubgraphKey, LatencyEstimator::Latency> id_profile;
  if (profile_database_json_["hash"].asUInt64() != GetProfileHash()) {
    BAND_LOG_INTERNAL(
        BAND_LOG_WARNING,
        "Current profile hash does not matches with a file (%s). Will ignore.",
        profile_data_path_.c_str());
    return id_profile;
  }

  for (auto profile_it = profile_database_json_.begin();
       profile_it != profile_database_json_.end(); ++profile_it) {
    std::string model_name = profile_it.key().asString();

    if (model_name != model_fname) {
      // We're only interested in `model_fname`.
      // NOTE: In case a model is using a different string name alias for
      // some other reason (e.g., two instances of the same model), we won't
      // be able to detect that the model can indeed reuse this profile.
      // An ad-hoc fix would be to add yet another "model name" field,
      // solely for profiling purposes.
      continue;
    }

    const Json::Value idx_profile = *profile_it;
    for (auto idx_profile_it = idx_profile.begin();
         idx_profile_it != idx_profile.end(); ++idx_profile_it) {
      std::string idx = idx_profile_it.key().asString();

      // parse the key to retrieve start/end indices
      // e.g., "25/50" --> delim_pos = 2
      auto delim_pos = idx.find("/");
      std::set<int> root_indices =
          string_to_node_indices(idx.substr(0, delim_pos));
      std::set<int> leaf_indices = string_to_node_indices(
          idx.substr(delim_pos + 1, idx.length() - delim_pos - 1));

      const Json::Value device_profile = *idx_profile_it;
      for (auto device_profile_it = device_profile.begin();
           device_profile_it != device_profile.end(); ++device_profile_it) {
        int worker_id = device_profile_it.key().asInt();
        int64_t profiled_latency = (*device_profile_it).asInt64();

        if (profiled_latency <= 0) {
          // jsoncpp treats missing values (null) as zero,
          // so they will be filtered out here
          continue;
        }

        SubgraphKey key(model_id, worker_id, root_indices, leaf_indices);
        id_profile[key] = {profiled_latency, profiled_latency};
      }
    }
  }
  return id_profile;
}

Json::Value LatencyEstimator::ProfileToJson() {
  Json::Value name_profile;
  name_profile["hash"] = GetProfileHash();
  for (auto& pair : profile_database_) {
    SubgraphKey key = pair.first;
    const int model_id = key.GetModelId();
    const std::string start_indices = key.GetInputOpsString();
    const std::string end_indices = key.GetOutputOpsString();
    const int64_t profiled_latency = pair.second.profiled;

    // check the string name of this model id
    auto model_spec = context_->GetModelSpec(model_id);
    if (model_spec && !model_spec->path.empty()) {
      // copy all entries in id_profile --> database_json
      // as an ad-hoc method, we simply concat the start/end indices to form
      // the level-two key in the final json value
      const std::string index_key = start_indices + "/" + end_indices;
      name_profile[model_spec->path][index_key][key.GetWorkerId()] =
          profiled_latency;
    } else {
      BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                        "Cannot find model %d from "
                        "model_configs. Will ignore.",
                        model_id);
      continue;
    }
  }
  return name_profile;
}

}  // namespace Band