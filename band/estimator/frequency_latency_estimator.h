#ifndef BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_
#define BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_

#include "band/common.h"
#include "band/config.h"
#include "resource_monitor.h"
#include "band/estimator/estimator_interface.h"

namespace band {

class FrequencyLatencyEstimator : public IEstimator<SubgraphKey> {
 public:
  explicit FrequencyLatencyEstimator(IEngine* engine, ResourceMonitor* monitor)
      : IEstimator(engine), resource_monitor_(monitor) {}
  absl::Status Init(const LatencyProfileConfig& config);
  void Update(const SubgraphKey& key, int64_t latency) override;

  absl::Status Load(ModelId model_id, std::string profile_path) override;
  absl::Status Profile(ModelId model_id) override;
  int64_t GetProfiled(const SubgraphKey& key) const override;
  int64_t GetExpected(const SubgraphKey& key) const override;

  absl::Status DumpProfile() override;

 private:
  std::string profile_path_;
  ResourceMonitor* resource_monitor_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_