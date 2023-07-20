#include "band/estimator/thermal_estimator.h"

namespace band {

absl::Status ThermalEstimator::Init(const ProfileConfig& config) {
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, int64_t new_value) {}

absl::Status ThermalEstimator::Profile(ModelId model_id) {
  return absl::OkStatus();
}

int64_t ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return 0;
}

int64_t ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  return 0;
}

int64_t ThermalEstimator::GetWorst(ModelId model_id) const {
  return 0;
}

absl::Status ThermalEstimator::DumpProfile() {
  return absl::OkStatus();
}

}  // namespace band