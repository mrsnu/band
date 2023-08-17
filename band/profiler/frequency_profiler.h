#ifndef BAND_PROFILER_FREQUECY_PROFILER_H_
#define BAND_PROFILER_FREQUECY_PROFILER_H_

#include <chrono>
#include <vector>

#include "band/config.h"
#include "band/device/frequency.h"
#include "band/profiler/profiler.h"

namespace band {

using FreqInfo = std::pair<std::chrono::system_clock::time_point, FreqMap>;

class FrequencyProfiler : public Profiler {
 public:
  explicit FrequencyProfiler(DeviceConfig config)
      : frequency_(new Frequency(config)) {}

  size_t BeginEvent() override;
  void EndEvent(size_t event_handle) override;
  size_t GetNumEvents() const override;

  FreqInfo GetFreqInfoStart(size_t index) const {
    if (timeline_.size() <= index) {
      return {};
    }
    return timeline_[index].first;
  }

  FreqInfo GetFreqInfoEnd(size_t index) const {
    if (timeline_.size() <= index) {
      return {};
    }
    return timeline_[index].second;
  }

 private:
  std::unique_ptr<Frequency> frequency_;
  std::vector<std::pair<FreqInfo, FreqInfo>> timeline_;
};

}  // namespace band

#endif  // BAND_PROFILER_FREQUECY_PROFILER_H_