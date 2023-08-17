#ifndef BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_
#define BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_

#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/profiler/frequency_profiler.h"
#include "band/profiler/latency_profiler.h"

namespace band {

class FrequencyLatencyEstimator : public IEstimator<SubgraphKey, FreqInfo> {
 public:
  explicit FrequencyLatencyEstimator(IEngine* engine,
                                     FrequencyProfiler* frequency_profiler,
                                     LatencyProfiler* latency_profiler)
      : IEstimator(engine) {}
  absl::Status Init(const LatencyProfileConfig& config) { return absl::OkStatus(); }
  void Update(const SubgraphKey& key, FreqInfo latency) override {}

  absl::Status Load(ModelId model_id, std::string profile_path) override {
    return absl::OkStatus();
  }
  absl::Status Profile(ModelId model_id) override { return absl::OkStatus(); }
  FreqInfo GetProfiled(const SubgraphKey& key) const override { return {}; }
  FreqInfo GetExpected(const SubgraphKey& key) const override { return {}; }

  absl::Status DumpProfile() override { return absl::OkStatus(); }

 private:
  FrequencyProfiler* frequency_profiler_;
  LatencyProfiler* latency_profiler_;
  std::string profile_path_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_