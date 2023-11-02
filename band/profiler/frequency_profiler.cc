#include "band/profiler/frequency_profiler.h"

#include <fstream>
#include <cerrno>

#include "band/logger.h"

namespace band {

FrequencyProfiler::FrequencyProfiler(DeviceConfig config)
    : frequency_(new Frequency(config)) {
  BAND_LOG_PROD(BAND_LOG_INFO, "FrequencyProfiler is created.");
}

size_t FrequencyProfiler::BeginEvent() {
  FreqInfo info = {std::chrono::system_clock::now(),
                   frequency_->GetAllFrequency()};
  timeline_.push_back({info, {}});
  return timeline_.size() - 1;
}

void FrequencyProfiler::EndEvent(size_t event_handle) {
  if (event_handle >= timeline_.size()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "Invalid event handle: %lu (timeline size: %lu)",
                  event_handle, timeline_.size());
    return;
  }
  FreqInfo info = {std::chrono::system_clock::now(),
                   frequency_->GetAllFrequency()};
  timeline_[event_handle].second = info;
}

size_t FrequencyProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band
