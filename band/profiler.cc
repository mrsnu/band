#include "band/profiler.h"

#include "band/json_util.h"

namespace Band {
absl::Status Profiler::Init(const ProfileConfig& config) {
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
}

void Profiler::UpdateLatency(const SubgraphKey& key, int64_t latency) {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    int64_t prev_latency = it->second.moving_averaged;
    profile_database_[key].moving_averaged =
        profile_smoothing_factor_ * latency +
        (1 - profile_smoothing_factor_) * prev_latency;
  }
}

int64_t Profiler::GetProfiled(const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second.profiled;
  } else {
    return -1;
  }
}

int64_t Profiler::GetExpected(const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second.moving_averaged;
  } else {
    return -1;
  }
}
}  // namespace Band