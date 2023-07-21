#include "band/estimator/network_estimator.h"

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler.h"
#include "band/worker.h"

namespace band {

absl::Status NetworkEstimator::Init(const ProfileConfig& config) {
  return absl::OkStatus();
}

void NetworkEstimator::Update(const SubgraphKey& key, int64_t latency) {
}

absl::Status NetworkEstimator::Profile(ModelId model_id) {
  return absl::OkStatus();
}

int64_t NetworkEstimator::GetProfiled(const SubgraphKey& key) const {
  return 0;
}

int64_t NetworkEstimator::GetExpected(const SubgraphKey& key) const {
  return 0;
}

int64_t NetworkEstimator::GetWorst(ModelId model_id) const {
  return 0;
}

absl::Status NetworkEstimator::DumpProfile() {
  return absl::OkStatus();
}

}  // namespace band