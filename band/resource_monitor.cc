#include "band/resource_monitor.h"

#include <iostream>
#include <dirent.h>

#include "band/logger.h"

namespace band {

namespace {

std::vector<std::string> ListFilesInPath(const char* path) {
  std::vector<std::string> ret;
  DIR* dir = opendir(path);
  if (dir == nullptr) {
    return {};
  }
  struct dirent* entry = readdir(dir);

  while (entry != nullptr) {
    if (entry->d_type == DT_REG) {
      ret.push_back(entry->d_name);
    }
    entry = readdir(dir);
  }
  closedir(dir);
  return ret;
}

std::vector<std::string> ListDirectoriesInPath(const char* path) {
  std::vector<std::string> ret;
  DIR* dir = opendir(path);
  if (dir == nullptr) {
    return {};
  }
  struct dirent* entry = readdir(dir);

  while (entry != nullptr) {
    if (entry->d_type == DT_DIR || entry->d_type == DT_LNK) {
      ret.push_back(entry->d_name);
    }
    entry = readdir(dir);
  }
  closedir(dir);
  return ret;
}

bool IsValidThermalZone(std::string path) {
  if (path.find("thermal_zone") == std::string::npos) {
    return false;
  }
  bool is_temp = false;
  auto contents = ListFilesInPath(path.c_str());
  for (auto& content : contents) {
    if (content == "temp") {
      is_temp = true;
    }
  }
  bool is_type = false;
  for (auto& content : contents) {
    if (content == "type") {
      is_type = true;
    }
  }
  return is_temp && is_type;
}

bool IsValidCpuFreq(std::string path) {
  auto contents = ListFilesInPath(path.c_str());
  for (auto& content : contents) {
    if (content == "cpuinfo_cur_freq") {
      return true;
    }
  }
  return false;
}

bool IsValidDevFreq(std::string path) {
  auto contents = ListFilesInPath(path.c_str());
  for (auto& content : contents) {
    if (content == "cur_freq") {
      return true;
    }
  }
  return false;
}

std::string MakeJsonList(std::vector<std::string> list) {
  std::string ret = "[";
  for (int i = 0; i < list.size(); i++) {
    ret += list[i];
    if (i != list.size() - 1) {
      ret += ", ";
    }
  }
  ret += "]";
  return ret;
}

}  // anonymous namespace

ResourceMonitor::~ResourceMonitor() {
  if (log_file_.is_open()) {
    log_file_.close();
  }
}

ResourceMonitor& ResourceMonitor::Create(std::string log_path) {
  static ResourceMonitor resource_monitor;
  auto status = resource_monitor.Init(log_path);
  if (!status.ok()) {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Cannot initialize resource monitor.");
  }
  return resource_monitor;
}

absl::Status ResourceMonitor::Init(std::string log_path) {
  auto tz_paths = GetThermalZonePaths();
  auto cpufreq_paths = GetCpuFreqPaths();
  auto devfreq_paths = GetDevFreqPaths();

  if (log_path.size() > 0) {
    log_file_.open(log_path, std::ios::out);
    if (!log_file_.is_open()) {
      return absl::InternalError("Cannot open log file.");
    }
  }

  for (auto& tz_path : tz_paths) {
    auto status = AddTemperatureResource(tz_path);
    if (!status.ok()) {
      return status;
    }
  }

  for (auto& cpufreq_path : cpufreq_paths) {
    auto status = AddCPUFrequencyResource(cpufreq_path);
    if (!status.ok()) {
      return status;
    }
  }

  for (auto& devfreq_path : devfreq_paths) {
    auto status = AddDevFrequencyResource(devfreq_path);
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<const Thermal> ResourceMonitor::GetCurrentThermal() const {
  Thermal ret;
  for (auto& tz : tzs_) {
    std::string type_path = tz + "/type";
    std::string temp_path = tz + "/temp";
    std::ifstream type_file(type_path);
    std::ifstream temp_file(temp_path);
    std::string type;
    int32_t temp;
    temp_file >> temp;
    type_file >> type;
    ret.status[type] = temp;
  }
  return ret;
}

absl::StatusOr<const Frequency> ResourceMonitor::GetCurrentFrequency() const {
  Frequency ret;
  for (auto& cpufreq : cpufreqs_) {
    std::string cpuinfo_cur_freq_path = cpufreq + "/cpuinfo_cur_freq";
    std::ifstream cpuinfo_cur_freq_file(cpuinfo_cur_freq_path);
    if (!cpuinfo_cur_freq_file.is_open()) {
      return absl::InternalError("Cannot open cpuinfo_cur_freq file");
    }
    int32_t cpuinfo_cur_freq;
    cpuinfo_cur_freq_file >> cpuinfo_cur_freq;
    ret.status[cpufreq] = cpuinfo_cur_freq;
  }
  for (auto& devfreq : devfreqs_) {
    std::string cur_freq_path = devfreq + "/cur_freq";
    std::ifstream cur_freq_file(cur_freq_path);
    if (!cur_freq_file.is_open()) {
      return absl::InternalError("Cannot open cur_freq file");
    }
    int32_t cur_freq;
    cur_freq_file >> cur_freq;
    ret.status[devfreq] = cur_freq;
  }
  return ret;
}

absl::StatusOr<const Network> ResourceMonitor::GetCurrentNetwork() const {
  return Network();
}

absl::StatusOr<std::tuple<const Thermal, const Frequency, const Network>>
ResourceMonitor::GetCurrentStatus() const {
  auto thermal = GetCurrentThermal();
  auto frequency = GetCurrentFrequency();
  auto network = GetCurrentNetwork();
  Thermal thermal_value;
  Frequency frequency_value;
  Network network_value;

  if (!thermal.ok()) {
    thermal_value = Thermal();
  } else {
    thermal_value = thermal.value();
  }
  if (!frequency.ok()) {
    frequency_value = Frequency();
  } else {
    frequency_value = frequency.value();
  }
  if (!network.ok()) {
    network_value = Network();
  } else {
    network_value = network.value();
  }
  if (log_file_.is_open()) {
    log_file_ << MakeJsonList({thermal_value.ToJson(),
                               frequency_value.ToJson(),
                               network_value.ToJson()}) << ",";
  }
  return std::make_tuple(thermal_value, frequency_value, network_value);
}

absl::Status ResourceMonitor::AddTemperatureResource(
    std::string resource_path) {
  if (!IsValidThermalZone(resource_path)) {
    return absl::InternalError("Invalid thermal zone path");
  }
  tzs_.push_back(resource_path);
  return absl::OkStatus();
}

absl::Status ResourceMonitor::AddCPUFrequencyResource(
    std::string resource_path) {
  if (!IsValidCpuFreq(resource_path)) {
    return absl::InternalError("Invalid cpu frequency path");
  }
  cpufreqs_.push_back(resource_path);
  return absl::OkStatus();
}

absl::Status ResourceMonitor::AddDevFrequencyResource(
    std::string resource_path) {
  if (!IsValidDevFreq(resource_path)) {
    return absl::InternalError("Invalid dev frequency path");
  }
  devfreqs_.push_back(resource_path);
  return absl::OkStatus();
}

absl::Status ResourceMonitor::AddNetworkResource(std::string resource_path) {
  return absl::OkStatus();
}

std::vector<std::string> ResourceMonitor::GetDetectedThermalZonePaths() const {
  return tzs_;
}

std::vector<std::string> ResourceMonitor::GetDetectedCpuFreqPaths() const {
  return cpufreqs_;
}

std::vector<std::string> ResourceMonitor::GetDetectedDevFreqPaths() const {
  return devfreqs_;
}

std::vector<std::string> ResourceMonitor::GetDetectedNetworkPaths() const {
  return {};
}

std::vector<std::string> ResourceMonitor::GetThermalZonePaths() const {
  std::vector<std::string> ret;
  auto thermal_zones = ListDirectoriesInPath(THERMAL_ZONE_BASE_PATH);
  for (auto& tz : thermal_zones) {
    auto thermal_zone_path = THERMAL_ZONE_BASE_PATH + tz;
    if (IsValidThermalZone(thermal_zone_path)) {
      ret.push_back(thermal_zone_path);
    }
  }
  return ret;
}

std::vector<std::string> ResourceMonitor::GetCpuFreqPaths() const {
  std::vector<std::string> ret;
  auto cpufreqs = ListDirectoriesInPath(CPUFREQ_BASE_PATH);
  for (auto& cpufreq : cpufreqs) {
    auto cpufreq_path = CPUFREQ_BASE_PATH + cpufreq;
    if (IsValidCpuFreq(cpufreq_path)) {
      ret.push_back(cpufreq_path);
    }
  }
  return ret;
}

std::vector<std::string> ResourceMonitor::GetDevFreqPaths() const {
  std::vector<std::string> ret;
  auto devfreqs = ListDirectoriesInPath(DEVFREQ_BASE_PATH);
  for (auto& devfreq : devfreqs) {
    auto devfreq_path = DEVFREQ_BASE_PATH + devfreq;
    if (IsValidDevFreq(devfreq_path)) {
      ret.push_back(devfreq_path);
    }
  }
  return ret;
}

}  // namespace band