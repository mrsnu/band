#ifndef BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_
#define BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>

#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"

#include "json/json.h"
#include "absl/status/status.h"

namespace band {

class ThermalEstimator : public IEstimator {
 public:
  explicit ThermalEstimator(IEngine* engine) : IEstimator(engine) {}
  absl::Status Init(const ProfileConfig& config) override;
  void Update(const SubgraphKey& key, int64_t new_value) override;
  absl::Status Profile(ModelId model_id) override;
  int64_t GetProfiled(const SubgraphKey& key) const override;
  int64_t GetExpected(const SubgraphKey& key) const override;
  int64_t GetWorst(ModelId model_id) const override;

  absl::Status DumpProfile() override;
  
};

}

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_