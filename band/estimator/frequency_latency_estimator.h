#ifndef BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_
#define BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_

#include "band/common.h"
#include "band/config.h"
#include "band/estimator/estimator_interface.h"
#include "band/profiler/frequency_profiler.h"
#include "band/profiler/latency_profiler.h"

namespace band {

class FrequencyLatencyEstimator
    : public IEstimator<SubgraphKey, double, double> {
 public:
  explicit FrequencyLatencyEstimator(IEngine* engine,
                                     FrequencyProfiler* frequency_profiler,
                                     LatencyProfiler* latency_profiler)
      : IEstimator(engine),
        frequency_profiler_(frequency_profiler),
        latency_profiler_(latency_profiler) {
    BAND_LOG_PROD(BAND_LOG_INFO, "FrequencyLatencyEstimator is created");
  }
  absl::Status Init(const FrequencyLatencyProfileConfig& config);
  void Update(const SubgraphKey& key, FreqInfo freq_info, double latency);
  void UpdateWithEvent(const SubgraphKey& key, size_t event_handle) override;

  double GetProfiled(const SubgraphKey& key) const override;
  double GetExpected(const SubgraphKey& key) const override;

  absl::Status LoadModel(std::string profile_path) override;
  absl::Status DumpModel(std::string profile_path) override;

 private:
  float profile_smoothing_factor_ = 0.1f;
  FrequencyProfiler* frequency_profiler_;
  LatencyProfiler* latency_profiler_;
  // SubgraphKey -> (frequency -> latency)
  std::map<SubgraphKey, std::map<double, double>> freq_lat_map_;
  std::map<SubgraphKey, double> profile_database_;
};

}  // namespace band

#endif  // BAND_ESTIMATOR_FREQUENCY_LATENCY_ESTIMATOR_H_