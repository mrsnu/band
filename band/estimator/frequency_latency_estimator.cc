#include "band/estimator/frequency_latency_estimator.h"

namespace band {

void FrequencyLatencyEstimator::UpdateWithEvent(const SubgraphKey& key,
                                                size_t event_handle) {
  Update(key, latency_profiler_->GetDuration(event_handle));
}
  
}  // namespace band
