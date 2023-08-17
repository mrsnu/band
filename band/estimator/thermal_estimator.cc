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

Eigen::VectorXd ConvertThermalMapToEigenVector(const ThermalMap& value) {
  Eigen::VectorXd vec(value.size());
  int i = 0;
  for (const auto& pair : value) {
    vec(i) = pair.second;
    i++;
  }
  return vec;
}

ThermalMap ConvertEigenVectorToThermalMap(const Eigen::VectorXd& vec) {
  ThermalMap value;
  for (int i = 0; i < vec.size(); i++) {
    value[static_cast<DeviceFlag>(i)] = vec(i);
  }
  return value;
}

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalInterval thermal) {
  Eigen::VectorXd old_vec = ConvertThermalMapToEigenVector(thermal.first.second);
  Eigen::VectorXd new_vec = ConvertThermalMapToEigenVector(thermal.second.second);
  ConvertEigenVectorToThermalMap(new_vec);
}

void ThermalEstimator::UpdateWithEvent(const SubgraphKey& key,
                                       size_t event_handle) {
  Update(key, thermal_profiler_->GetInterval(event_handle));
}

absl::Status ThermalEstimator::LoadProfile(std::string profile_path) {
  return absl::OkStatus();
}

ThermalMap ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return {};
}

ThermalMap ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  return {};
}

absl::Status ThermalEstimator::DumpProfile(std::string path) {
  return absl::OkStatus();
}

}  // namespace band