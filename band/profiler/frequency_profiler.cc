#include "band/profiler/frequency_profiler.h"

#include "band/logger.h"

namespace band {

namespace {

std::string FreqInfoToString(const FreqInfo& info) {
  std::string result = "{";
  // time
  result += "\"time\":";
  result +=
      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                         info.first.time_since_epoch())
                         .count());
  result += ",";
  // frequency
  result += "\"frequency\":{";
  for (auto& pair : info.second) {
    result += "\"" + std::string(ToString(pair.first)) + "\":";
    result += std::to_string(pair.second);
    result += ",";
  }
  result.pop_back();
  result += "}";
  result += "}";

  return result;
}

}  // anonymous namespace

FrequencyProfiler::FrequencyProfiler(DeviceConfig config)
    : frequency_(new Frequency(config)) {
  BAND_LOG_PROD(BAND_LOG_INFO, "FrequencyProfiler is created.");
  log_file_.open(config.freq_log_path, std::ios::out);
  if (!log_file_.is_open()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "FrequencyProfiler failed to open the log file %s",
                  config.freq_log_path.c_str());
    return;
  }
  log_file_ << "{";
}

FrequencyProfiler::~FrequencyProfiler() {
  if (log_file_.is_open()) {
    log_file_.seekp(-1, std::ios_base::end);
    log_file_ << "}";
    log_file_.close();
  }
}

size_t FrequencyProfiler::BeginEvent() {
  FreqInfo info = {std::chrono::system_clock::now(),
                   frequency_->GetAllFrequency()};
  log_file_ << FreqInfoToString(info) << ",";
  timeline_.push_back({info, {}});
  return timeline_.size() - 1;
}

void FrequencyProfiler::EndEvent(size_t event_handle) {
  if (!event_handle || event_handle >= timeline_.size()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "Invalid event handle: %lu (timeline size: %lu)",
                  event_handle, timeline_.size());
    return;
  }
  FreqInfo info = {std::chrono::system_clock::now(),
                   frequency_->GetAllFrequency()};
  log_file_ << FreqInfoToString(info) << ",";
  timeline_[event_handle - 1].second = info;
}

size_t FrequencyProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band
