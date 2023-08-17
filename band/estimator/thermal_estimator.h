#ifndef BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_
#define BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_

#include <chrono>
#include <deque>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/profiler/thermal_profiler.h"
#include "json/json.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"

namespace band {

class ThermalEstimator : public IEstimator<SubgraphKey, ThermalInfo> {
 public:
  explicit ThermalEstimator(IEngine* engine, ThermalProfiler* thermal_profiler)
      : IEstimator(engine), thermal_profiler_(thermal_profiler) {}
  absl::Status Init(const ThermalProfileConfig& config);
  void Update(const SubgraphKey& key, ThermalInfo thermal) override;
  void Update(const SubgraphKey& key, ThermalInfo old_value,
              ThermalInfo new_value);

  absl::Status Load(ModelId model_id, std::string profile_path) override;
  absl::Status Profile(ModelId model_id) override { return absl::OkStatus(); }
  ThermalInfo GetProfiled(const SubgraphKey& key) const override;
  ThermalInfo GetExpected(const SubgraphKey& key) const override;

  absl::Status DumpProfile() override;

 private:
  Eigen::MatrixXd SolveLinear(Eigen::MatrixXd x, Eigen::VectorXd y) {
    return (x.transpose() * x).ldlt().solve(x.transpose() * y);
  }

  ThermalProfiler* thermal_profiler_;

  size_t num_resources_ = 0;
  std::string profile_path_;
  Eigen::MatrixXd model_;

  std::deque<Eigen::VectorXd> history_queue_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_