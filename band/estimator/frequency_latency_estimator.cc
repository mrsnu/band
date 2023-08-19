#include "band/estimator/frequency_latency_estimator.h"

namespace band {

void FrequencyLatencyEstimator::UpdateWithEvent(const SubgraphKey& key,
                                                size_t event_handle) {
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency = latency_profiler_->GetDuration(event_handle);
  Update(key, freq_interval.second, latency);
}
  
}  // namespace band
