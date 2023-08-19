#include "band/estimator/thermal_estimator.h"

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler/profiler.h"
#include "band/worker.h"

namespace band {

namespace {

template <typename T>
Eigen::VectorXd ConvertTMapToEigenVector(const T& value, size_t size) {
  Eigen::VectorXd vec(size);
  int i = 0;
  for (const auto& pair : value) {
    vec(i) = pair.second;
    i++;
  }
  return vec;
}

template <typename T>
T ConvertEigenVectorToTMap(const Eigen::VectorXd& vec) {
  T value;
  for (int i = 0; i < vec.size(); i++) {
    if (vec(i) == 0) {
      value[static_cast<DeviceFlag>(i)] = vec(i);
    }
  }
  return value;
}

Eigen::VectorXd GetOneHotVector(double value, size_t size, size_t index) {
  Eigen::VectorXd vec(size);
  vec(index) = value;
  return vec;
}

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  window_size_ = config.window_size;
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalMap therm_start,
                              ThermalMap therm_end, FreqMap freq,
                              double latency) {
  const size_t num_devices = EnumLength<DeviceFlag>();
  Eigen::VectorXd old_therm =
      ConvertTMapToEigenVector<ThermalMap>(therm_start, num_devices);
  Eigen::VectorXd new_therm =
      ConvertTMapToEigenVector<ThermalMap>(therm_end, num_devices);
  Eigen::VectorXd freq_info =
      ConvertTMapToEigenVector<FreqMap>(freq, num_devices);
  Eigen::VectorXd latency_vector = GetOneHotVector(
      latency, num_devices,
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));

  // num_devices + num_devices + num_devices
  size_t feature_size = old_therm.size() + freq_info.size() +
                        latency_vector.size() + latency_vector.size();
  size_t target_size = new_therm.size();

  Eigen::VectorXd feature(feature_size);
  feature << old_therm, freq_info, (freq_info * latency_vector), latency_vector;

  features_.push_back({feature, new_therm});
  if (features_.size() > window_size_) {
    features_.pop_front();
  }
  if (features_.size() < window_size_) {
    BAND_LOG_PROD(BAND_LOG_INFO,
                  "ThermalEstimator, Not enough data collected. Current number "
                  "of data: %d",
                  features_.size());
    return;
  }

  Eigen::MatrixXd data(window_size_, feature_size);
  Eigen::MatrixXd target(window_size_, target_size);
  for (int i = 0; i < window_size_; i++) {
    data << features_[i].first;
    target << features_[i].second;
  }
  model_ = SolveLinear(data, target);
}

void ThermalEstimator::UpdateWithEvent(const SubgraphKey& key,
                                       size_t event_handle) {
  auto therm_interval = thermal_profiler_->GetInterval(event_handle);
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency = latency_profiler_->GetDuration(event_handle);
  Update(key, therm_interval.first.second, therm_interval.second.second,
         freq_interval.second.second, latency);
}

ThermalMap ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return ConvertEigenVectorToTMap<ThermalMap>(
      features_[features_.size() - 1].second);
}

ThermalMap ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  const size_t num_devices = EnumLength<DeviceFlag>();
  auto cur_therm = ConvertTMapToEigenVector<ThermalMap>(
      thermal_profiler_->GetAllThermal(), num_devices);
  return ConvertEigenVectorToTMap<ThermalMap>(model_.transpose() * cur_therm);
}

absl::Status ThermalEstimator::LoadModel(std::string profile_path) {
  return absl::OkStatus();
}

absl::Status ThermalEstimator::DumpModel(std::string profile_path) {
  return absl::OkStatus();
}

}  // namespace band