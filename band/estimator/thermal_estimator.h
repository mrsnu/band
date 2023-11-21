#ifndef BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_
#define BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <unordered_map>

#include "absl/status/status.h"
#include "band/common.h"
#include "band/config.h"
#include "band/device/frequency.h"
#include "band/device/thermal.h"
#include "band/estimator/estimator_interface.h"
#include "band/profiler/thermal_profiler.h"
#include "json/json.h"
#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"

namespace band {

using ThermalKey = std::tuple<SubgraphKey, ThermalMap, FreqMap>;

class ThermalEstimator
    : public IEstimator<ThermalKey, ThermalInterval, ThermalMap> {
 public:
  explicit ThermalEstimator(
      IEngine* engine,
      IEstimator<SubgraphKey, double, double>* latency_estimator)
      : IEstimator(engine),
        latency_estimator_(latency_estimator),
        model_update_thread_(
            std::thread(&ThermalEstimator::ModelUpdateThreadLoop, this)) {}
  ~ThermalEstimator();
  absl::Status Init(const ThermalProfileConfig& config);

  void Update(const SubgraphKey& key, Job& job) override;

  ThermalMap GetProfiled(const SubgraphKey& key) const override;
  ThermalMap GetExpected(const ThermalKey& thermal_key) const override;

  absl::Status LoadModel(std::string profile_path) override;
  absl::Status DumpModel(std::string profile_path) override;

  void UpdateModel();

  Json::Value EigenMatrixToJson(Eigen::MatrixXd& matrix);

  Eigen::MatrixXd JsonToEigenMatrix(Json::Value json);

 private:
  Eigen::VectorXd GetFeatureVector(const Eigen::VectorXd& therm_vec,
                                   const Eigen::VectorXd& freq_vec,
                                   const Eigen::VectorXd& lat_vec,
                                   size_t worker_id, double latency) const;

  IEstimator<SubgraphKey, double, double>* latency_estimator_;

  size_t num_resources_ = 0;
  size_t window_size_;

  // WorkerId, start_time, end_time, start_therm, end_therm, freq
  typedef std::tuple<WorkerId, int64_t, int64_t, Eigen::VectorXd,
                     Eigen::VectorXd, Eigen::VectorXd>
      Feature;

  std::mutex model_mutex_;
  Eigen::MatrixXd model_;
  std::deque<Feature> features_;

  mutable std::map<SubgraphKey, ThermalMap> profile_database_;

  const size_t num_sensors_ = EnumLength<SensorFlag>();
  const size_t num_devices_ = EnumLength<FreqFlag>();
  const size_t feature_size_ = num_sensors_ + 3 * num_devices_;

  bool model_update_thread_exit_ = false;
  std::condition_variable model_update_cv_;
  std::thread model_update_thread_;
  std::mutex model_update_queue_mutex_;
  std::queue<Feature> model_update_queue_;

  void ModelUpdateThreadLoop();
};

}  // namespace band

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_