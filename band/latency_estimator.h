#ifndef BAND_LATENCY_ESTIMATOR_H_
#define BAND_LATENCY_ESTIMATOR_H_

#include <json/json.h>

#include <chrono>
#include <map>

#include "band/common.h"
#include "band/config.h"

namespace Band {
class Context;
class LatencyEstimator {
 public:
  explicit LatencyEstimator(Context* context);
  BandStatus Init(const ProfileConfig& config);
  void UpdateLatency(const SubgraphKey& key, int64_t latency);

  BandStatus ProfileModel(ModelId model_id);
  int64_t GetProfiled(const SubgraphKey& key) const;
  int64_t GetExpected(const SubgraphKey& key) const;

 private:
  // latency in microseconds
  struct Latency {
    int64_t profiled;
    int64_t moving_averaged;
  };

  // Path to the profile data.
  // The data in the path will be read during initial phase, and also
  // will be updated at the end of the run.
  std::string profile_data_path_;

  // The contents of the file at `profile_data_path_`.
  // We keep this separately from `profile_database_`, since we cannot
  // immediately put `profile_data_path_`'s contents into `profile_database_`
  // because the model name --> int mapping is not available at init time.
  Json::Value profile_database_json_;

  std::map<SubgraphKey, Latency> profile_database_;
  float profile_smoothing_factor_;

  bool profile_online_;
  int profile_num_warmups_;
  int profile_num_runs_;
  std::vector<int> profile_copy_computation_ratio_;

  Context* const context_;
};
}  // namespace Band

#endif  // BAND_LATENCY_ESTIMATOR_H_