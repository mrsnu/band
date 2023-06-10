#ifndef BAND_LATENCY_ESTIMATOR_H_
#define BAND_LATENCY_ESTIMATOR_H_

#include <json/json.h>

#include <chrono>
#include <unordered_map>

#include "band/common.h"
#include "band/config.h"

namespace band {
class IEngine;
class LatencyEstimator {
 public:
  explicit LatencyEstimator(IEngine* engine);
  absl::Status Init(const ProfileConfig& config);
  void UpdateLatency(const SubgraphKey& key, int64_t latency);

  absl::Status ProfileModel(ModelId model_id);
  int64_t GetProfiled(const SubgraphKey& key) const;
  int64_t GetExpected(const SubgraphKey& key) const;
  int64_t GetWorst(ModelId model_id) const;

  absl::Status DumpProfile();

  // latency in microseconds
  struct Latency {
    int64_t profiled;
    int64_t moving_averaged;
  };

 private:
  size_t GetProfileHash() const;

  // Convert entries in the json value to ModelDeviceToLatency format,
  // for the given model name and target model id.
  std::map<SubgraphKey, Latency> JsonToModelProfile(
      const std::string& model_fname, const int model_id);

  // Convert model integer ids back to string-type names for model profiles,
  // and returns the json format identical to `profile_database_json_`.
  Json::Value ProfileToJson();

  // Path to the profile data.
  // The data in the path will be read during initial phase, and also
  // will be updated at the end of the run.
  std::string profile_data_path_;

  // The contents of the file at `profile_data_path_`.
  // We keep this separately from `profile_database_`, since we cannot
  // immediately put `profile_data_path_`'s contents into `profile_database_`
  // because the model name --> int mapping is not available at init time.
  Json::Value profile_database_json_;

  std::unordered_map<SubgraphKey, Latency, SubgraphHash> profile_database_;
  float profile_smoothing_factor_ = 0.05f;

  bool profile_online_;
  int profile_num_warmups_;
  int profile_num_runs_;
  std::vector<int> profile_copy_computation_ratio_;

  IEngine* const engine_;
};
}  // namespace band

#endif  // BAND_LATENCY_ESTIMATOR_H_