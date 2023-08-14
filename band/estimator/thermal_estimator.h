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

typedef std::pair<SubgraphKey, int> ThermalSubgraphKey;

class ThermalEstimator : public IEstimator<ThermalSubgraphKey> {
 public:
  explicit ThermalEstimator(IEngine* engine) : IEstimator(engine) {}
  absl::Status Init(const ThermalProfileConfig& config);
  void Update(const ThermalSubgraphKey& key, int64_t latency) override;

  absl::Status Profile(ModelId model_id) override;
  int64_t GetProfiled(const ThermalSubgraphKey& key) const override;
  int64_t GetExpected(const ThermalSubgraphKey& key) const override;

  absl::Status DumpProfile() override;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_THERMAL_ESTIMATOR_H_