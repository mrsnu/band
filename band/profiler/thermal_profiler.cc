#include "band/profiler/thermal_profiler.h"

#include <fstream>
#include <cerrno>

#include "band/logger.h"

namespace band {

namespace {

std::string ThermalInfoToString(const ThermalInfo& info) {
  std::string result = "{";
  // time
  result += "\"time\":";
  result +=
      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                         info.first.time_since_epoch())
                         .count());
  result += ",";
  // thermal
  result += "\"thermal\":{";
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

}  // namespace

ThermalProfiler::ThermalProfiler(DeviceConfig config)
    : thermal_(new Thermal(config)) {
  BAND_LOG_PROD(BAND_LOG_INFO, "ThermalProfiler is created.");
  log_file_.open(config.therm_log_path, std::ios::out);
  if (!log_file_.is_open()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "ThermalProfiler failed to open the log file %s: %s",
                  config.therm_log_path.c_str(), strerror(errno));
    return;
  }
  log_file_ << "[";
}

ThermalProfiler::~ThermalProfiler() {
  if (log_file_.is_open()) {
    log_file_.seekp(-1, std::ios_base::end);
    log_file_ << "]";
    log_file_.close();
  }
}

size_t ThermalProfiler::BeginEvent() {
  ThermalInfo info = {std::chrono::system_clock::now(),
                      thermal_->GetAllThermal()};
  log_file_ << ThermalInfoToString(info) << ",";
  timeline_.push_back({info, {}});
  return timeline_.size() - 1;
}

void ThermalProfiler::EndEvent(size_t event_handle) {
  if (!event_handle || event_handle >= timeline_.size()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "ThermalProfiler end event with an invalid handle %d",
                  event_handle);
    return;
  }
  ThermalInfo info = {std::chrono::system_clock::now(),
                                        thermal_->GetAllThermal()};
  log_file_ << ThermalInfoToString(info) << ",";
  timeline_[event_handle - 1].second = info;
}

size_t ThermalProfiler::GetNumEvents() const { return timeline_.size(); }

}  // namespace band