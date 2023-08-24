#include "band/estimator/frequency_latency_estimator.h"

#include "band/engine_interface.h"
#include "band/json_util.h"

namespace band {

absl::Status FrequencyLatencyEstimator::Init(
    const FrequencyLatencyProfileConfig& config) {
  profile_smoothing_factor_ = config.smoothing_factor;
  return absl::OkStatus();
}

void FrequencyLatencyEstimator::Update(const SubgraphKey& key,
                                       FreqInfo freq_info, double latency) {
  profile_database_[key] = latency;
  auto device = engine_->GetWorkerDevice(key.GetWorkerId());
  auto freq = freq_info.second[device];
  auto it = freq_lat_map_.find(key);
  if (it == freq_lat_map_.end()) {
    auto avail_freqs_device =
        frequency_profiler_->GetAllAvailableFrequency()[device];
    freq_lat_map_[key] = std::map<double, double>();
    freq_lat_map_[key][0] = latency;
    for (auto& available_freq : avail_freqs_device) {
      freq_lat_map_[key][available_freq] = latency;
    }
    return;
  }
  double prev_latency = it->second[freq];
  freq_lat_map_[key][freq] = profile_smoothing_factor_ * latency +
                             (1 - profile_smoothing_factor_) * prev_latency;
}

void FrequencyLatencyEstimator::UpdateWithEvent(const SubgraphKey& key,
                                                size_t event_handle) {
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency =
      latency_profiler_->GetDuration<std::chrono::milliseconds>(event_handle);
  Update(key, freq_interval.first, latency);
}

double FrequencyLatencyEstimator::GetProfiled(const SubgraphKey& key) const {
  if (profile_database_.find(key) == profile_database_.end()) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "[GetProfiled] No profiled latency for %s.",
                  key.ToString().c_str());
    return 0;
  }
  return profile_database_.at(key);
}

double FrequencyLatencyEstimator::GetExpected(const SubgraphKey& key) const {
  auto freq =
      frequency_profiler_
          ->GetAllFrequency()[engine_->GetWorkerDevice(key.GetWorkerId())];
  auto it = freq_lat_map_.find(key);
  if (it != freq_lat_map_.end()) {
    auto it2 = it->second.find(freq);
    if (it2 != it->second.end()) {
      return it2->second;
    }
  }
  BAND_LOG_PROD(BAND_LOG_ERROR,
                "[GetExpected] No expected latency for %s with "
                "frequency %f.",
                key.ToString().c_str(), freq);
  return GetProfiled(key);
}

absl::Status FrequencyLatencyEstimator::LoadModel(std::string profile_path) {
  return absl::OkStatus();
}

absl::Status FrequencyLatencyEstimator::DumpModel(std::string profile_path) {
  Json::Value root;
  for (auto it = freq_lat_map_.begin(); it != freq_lat_map_.end(); ++it) {
    Json::Value key;
    key["model_id"] = it->first.GetModelId();
    key["worker_id"] = it->first.GetWorkerId();
    key["unit_indices"] = it->first.GetUnitIndicesString();
    for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      Json::Value freq;
      freq["frequency"] = it2->first;
      freq["latency"] = it2->second;
      key["frequency_latency"].append(freq);
    }
    root["subgraph"].append(key);
  }
  std::ofstream file(profile_path);
  file << root;
  return absl::OkStatus();
}

}  // namespace band
