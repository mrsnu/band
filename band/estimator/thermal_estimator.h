#ifndef BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_
#define BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_

#include <chrono>
#include <deque>
#include <unordered_map>
#include <tuple>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/estimator/frequency_latency_estimator.h"
#include "band/profiler/frequency_profiler.h"
#include "band/profiler/latency_profiler.h"
#include "band/profiler/thermal_profiler.h"
#include "json/json.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"

namespace band {

using ThermalKey = std::tuple<ThermalMap, FreqMap, SubgraphKey>;

class ThermalEstimator
    : public IEstimator<ThermalKey, ThermalInterval, ThermalMap> {
 public:
  explicit ThermalEstimator(IEngine* engine, ThermalProfiler* thermal_profiler,
                            FrequencyProfiler* frequency_profiler,
                            LatencyProfiler* latency_profiler,
                            IEstimator<SubgraphKey, double, double>* latency_estimator)
      : IEstimator(engine),
        thermal_profiler_(thermal_profiler),
        frequency_profiler_(frequency_profiler),
        latency_profiler_(latency_profiler),
        latency_estimator_(latency_estimator) {}
  absl::Status Init(const ThermalProfileConfig& config);
  void Update(const ThermalKey& key, ThermalMap target_therm);
  void UpdateWithEvent(const SubgraphKey& key, size_t event_handle) override;

  ThermalMap GetProfiled(const SubgraphKey& key) const override;
  ThermalMap GetExpected(const ThermalKey& thermal_key) const override;

  absl::Status LoadModel(std::string profile_path) override;
  absl::Status DumpModel(std::string profile_path) override;

  Eigen::MatrixXd SolveLinear(Eigen::MatrixXd& x, Eigen::MatrixXd& y);

  Json::Value EigenMatrixToJson(Eigen::MatrixXd& matrix);

  Eigen::MatrixXd JsonToEigenMatrix(Json::Value json);

 private:
  ThermalProfiler* thermal_profiler_;
  FrequencyProfiler* frequency_profiler_;
  LatencyProfiler* latency_profiler_;
  IEstimator<SubgraphKey, double, double>* latency_estimator_;

  size_t num_resources_ = 0;
  size_t window_size_;

  Eigen::MatrixXd model_;
  std::deque<std::pair<Eigen::VectorXd, Eigen::VectorXd>> features_;
  mutable std::map<SubgraphKey, ThermalMap> profile_database_;

  const size_t num_sensors_ = EnumLength<SensorFlag>();
  const size_t num_devices_ = EnumLength<DeviceFlag>();
  const size_t feature_size_ = num_sensors_ + 3 * num_devices_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_