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

Eigen::VectorXd ConvertThermalInfoToEigenVector(
    const band::ThermalInfo& value) {
  Eigen::VectorXd vec(value.size());
  int i = 0;
  for (const auto& pair : value) {
    vec(i) = pair.second;
    i++;
  }
  return vec;
}

ThermalInfo ConvertEigenVectorToThermalInfo(const Eigen::VectorXd& vec) {
  ThermalInfo value;
  for (int i = 0; i < vec.size(); i++) {
    value[static_cast<DeviceFlag>(i)] = vec(i);
  }
  return value;
}

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalInfo thermal) {}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalInfo old_value,
                              ThermalInfo new_value) {
  Eigen::VectorXd old_vec = ConvertThermalInfoToEigenVector(old_value);
  Eigen::VectorXd new_vec = ConvertThermalInfoToEigenVector(new_value);
  ConvertEigenVectorToThermalInfo(new_vec);
}

absl::Status ThermalEstimator::Load(ModelId model_id,
                                    std::string profile_path) {
  return absl::OkStatus();
}

ThermalInfo ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return {};
}

ThermalInfo ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  return {};
}

absl::Status ThermalEstimator::DumpProfile() { return absl::OkStatus(); }

}  // namespace band