#ifndef BAND_ESTIMATOR_LATENCY_ESTIMATOR_H_
#define BAND_ESTIMATOR_LATENCY_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/profiler/latency_profiler.h"
#include "json/json.h"

namespace band {

class IEngine;
class LatencyEstimator : public IEstimator<SubgraphKey, double, double> {
 public:
  explicit LatencyEstimator(IEngine* engine, LatencyProfiler* latency_profiler)
      : IEstimator(engine), latency_profiler_(latency_profiler) {}
  absl::Status Init(const LatencyProfileConfig& config);

  void Update(const SubgraphKey& key, double latency);
  void UpdateWithEvent(const SubgraphKey& key, size_t event_handle) override;
  double GetProfiled(const SubgraphKey& key) const override;
  double GetExpected(const SubgraphKey& key) const override;

  absl::Status LoadModel(std::string profile_path) override;
  absl::Status DumpModel(std::string profile_path) override;

  // latency in microseconds
  struct Latency {
    double profiled;
    double moving_averaged;
  };

 private:
  size_t GetProfileHash() const;

  absl::Status LoadFromProfiler();
  absl::Status LoadFromFile(std::string profile_path);

  // Convert entries in the json value to ModelDeviceToLatency format,
  // for the given model name and target model id.
  std::map<SubgraphKey, Latency> JsonToModelProfile(
      const std::string& model_fname, const int model_id);

  // Convert model integer ids back to string-type names for model profiles,
  // and returns the json format identical to `profile_database_json_`.
  Json::Value ProfileToJson();

  LatencyProfiler* latency_profiler_;

  // Path to the profile data.
  // The data in the path will be read during initial phase, and also
  // will be updated at the end of the run.
  std::string profile_path_;

  // The contents of the file at `latency_profile_path_`.
  // We keep this separately from `profile_database_`, since we cannot
  // immediately put `latency_profile_path_`'s contents into `profile_database_`
  // because the model name --> int mapping is not available at init time.
  Json::Value profile_database_json_;

  std::unordered_map<SubgraphKey, Latency, SubgraphHash> profile_database_;

  float profile_smoothing_factor_ = 0.1f;
  int profile_num_warmups_ = 1;
  int profile_num_runs_ = 1;
};
}  // namespace band

#endif  // BAND_ESTIMATOR_LATENCY_ESTIMATOR_H_