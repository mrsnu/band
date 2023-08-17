#ifndef BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_
#define BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_

#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/profiler/frequency_profiler.h"
#include "band/profiler/latency_profiler.h"

namespace band {

class FrequencyLatencyEstimator : public IEstimator<SubgraphKey, int64_t> {
 public:
  explicit FrequencyLatencyEstimator(IEngine* engine,
                                     FrequencyProfiler* frequency_profiler,
                                     LatencyProfiler* latency_profiler)
      : IEstimator(engine),
        frequency_profiler_(frequency_profiler),
        latency_profiler_(latency_profiler) {}
  absl::Status Init(const FrequencyLatencyProfileConfig& config) {
    return absl::OkStatus();
  }
  void Update(const SubgraphKey& key, int64_t latency) override {}
  void UpdateWithEvent(const SubgraphKey& key, size_t event_handle) override;
  
  int64_t GetProfiled(const SubgraphKey& key) const override { return {}; }
  int64_t GetExpected(const SubgraphKey& key) const override { return {}; }

  absl::Status LoadProfile(std::string profile_path) override {
    return absl::OkStatus();
  }
  absl::Status DumpProfile(std::string path) override {
    return absl::OkStatus();
  }

 private:
  FrequencyProfiler* frequency_profiler_;
  LatencyProfiler* latency_profiler_;
  std::string profile_path_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_