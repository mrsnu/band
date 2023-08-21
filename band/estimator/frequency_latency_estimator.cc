#include "band/estimator/frequency_latency_estimator.h"

#include "band/engine_interface.h"

namespace band {

absl::Status FrequencyLatencyEstimator::Init(
    const FrequencyLatencyProfileConfig& config) {
  profile_smoothing_factor_ = config.smoothing_factor;
  return absl::OkStatus();
}

void FrequencyLatencyEstimator::Update(const SubgraphKey& key,
                                       FreqInfo freq_info, double latency) {
  auto freq =
      frequency_profiler_
          ->GetAllFrequency()[engine_->GetWorkerDevice(key.GetWorkerId())];
  auto it = profile_database_.find(key);
  if (it == profile_database_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Initial profiled latency %s: %f.",
                      key.ToString().c_str(), latency);
    profile_database_[key] = {{freq, latency}};
    return;
  }
  if (it->second.find(freq) == it->second.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Initial profiled latency %s: %f.",
                      key.ToString().c_str(), latency);
    profile_database_[key][freq] = latency;
    return;
  }
  double prev_latency = it->second[freq];

  profile_database_[key][freq] = profile_smoothing_factor_ * latency +
                                 (1 - profile_smoothing_factor_) * prev_latency;
}

void FrequencyLatencyEstimator::UpdateWithEvent(const SubgraphKey& key,
                                                size_t event_handle) {
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency =
      latency_profiler_->GetDuration<std::chrono::milliseconds>(event_handle);
  Update(key, freq_interval.second, latency);
}

double FrequencyLatencyEstimator::GetProfiled(const SubgraphKey& key) const {
  return 0;
}

double FrequencyLatencyEstimator::GetExpected(const SubgraphKey& key) const {
  auto freq =
      frequency_profiler_
          ->GetAllFrequency()[engine_->GetWorkerDevice(key.GetWorkerId())];
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    auto it2 = it->second.find(freq);
    if (it2 != it->second.end()) {
      return it2->second;
    }
  }
  return 0;
}

absl::Status FrequencyLatencyEstimator::LoadModel(std::string profile_path) {
  return absl::OkStatus();
}

absl::Status FrequencyLatencyEstimator::DumpModel(std::string profile_path) {
  return absl::OkStatus();
}

}  // namespace band
