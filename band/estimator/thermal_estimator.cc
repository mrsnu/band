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
Eigen::VectorXd ConvertTMapToEigenVector(const T& value) {
  Eigen::VectorXd vec(value.size());
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
    value[static_cast<DeviceFlag>(i)] = vec(i);
  }
  return value;
}

Eigen::VectorXd GetVector(double value, size_t size, size_t index) {
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
  Eigen::VectorXd old_therm = ConvertTMapToEigenVector<ThermalMap>(therm_start);
  Eigen::VectorXd new_therm = ConvertTMapToEigenVector<ThermalMap>(therm_end);
  Eigen::VectorXd freq_info = ConvertTMapToEigenVector<FreqMap>(freq);
  WorkerId worker_id = key.GetWorkerId();
  Eigen::VectorXd latency_vector = GetVector(
      latency, EnumLength<DeviceFlag>(),
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));
  ConvertEigenVectorToTMap<ThermalMap>(old_therm);
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
  return {};
}

ThermalMap ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  return {};
}

absl::Status ThermalEstimator::LoadModel(std::string profile_path) {
  return absl::OkStatus();
}

absl::Status ThermalEstimator::DumpModel(std::string profile_path) {
  return absl::OkStatus();
}

}  // namespace band