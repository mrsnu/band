#ifndef BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_
#define BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>
#include <deque>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/resource_monitor.h"
#include "json/json.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"

namespace band {

using ThermalValue = std::map<DeviceFlag, int>;

class ThermalEstimator : public IEstimator<SubgraphKey, ThermalValue> {
 public:
  explicit ThermalEstimator(IEngine* engine, ResourceMonitor* monitor)
      : IEstimator(engine), resource_monitor_(monitor) {}
  absl::Status Init(const ThermalProfileConfig& config);
  void Update(const SubgraphKey& key, ThermalValue thermal) override;
  void Update(const SubgraphKey& key, ThermalValue old_value,
              ThermalValue new_value);

  absl::Status Load(ModelId model_id, std::string profile_path) override;
  absl::Status Profile(ModelId model_id) override;
  ThermalValue GetProfiled(const SubgraphKey& key) const override;
  ThermalValue GetExpected(const SubgraphKey& key) const override;

  absl::Status DumpProfile() override;

 private:
  Eigen::MatrixXd SolveLinear(Eigen::MatrixXd x, Eigen::VectorXd y) {
    return (x.transpose() * x).ldlt().solve(x.transpose() * y);
  }

  size_t num_resources_ = 0;
  std::string profile_path_;
  ResourceMonitor* resource_monitor_;
  Eigen::MatrixXd model_;

  std::deque<Eigen::VectorXd> history_queue_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_