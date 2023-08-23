#include "band/estimator/thermal_estimator.h"

#include <fstream>

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler/profiler.h"
#include "band/worker.h"

namespace band {

namespace {

template <typename T>
Eigen::VectorXd ConvertTMapToEigenVector(const T& value, size_t size) {
  Eigen::VectorXd vec(size);
  vec.setZero();
  for (const auto& pair : value) {
    vec(static_cast<size_t>(pair.first)) = pair.second;
  }
  return vec;
}

template <typename T>
T ConvertEigenVectorToTMap(const Eigen::VectorXd& vec) {
  T value;
  for (int i = 0; i < vec.size(); i++) {
    if (vec(i) != 0) {
      value[static_cast<SensorFlag>(i)] = vec(i);
    }
  }
  return value;
}

Eigen::VectorXd GetOneHotVector(double value, size_t size, size_t index) {
  Eigen::VectorXd vec(size);
  vec.setZero();
  vec(index) = value;
  return vec;
}

Eigen::VectorXd GetFillVector(double value, size_t size) {
  Eigen::VectorXd vec(size);
  for (int i = 0; i < size; i++) {
    vec(i) = value;
  }
  return vec;
}

std::string ThermalMapToJson(ThermalMap thermal_map) {
  std::string result = "{";
  for (auto& pair : thermal_map) {
    result += "\"" + std::string(ToString(pair.first)) + "\":";
    result += std::to_string(pair.second);
    result += ",";
  }
  result.pop_back();
  result += "}";
  return result;
}

void PrintVector(std::string name, const Eigen::VectorXd& vec) {
  BAND_LOG_PROD(BAND_LOG_INFO, "%s: ", name.c_str());
  for (int i = 0; i < vec.size(); i++) {
    BAND_LOG_PROD(BAND_LOG_INFO, "%f ", vec(i));
  }
}

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config,
                                    std::string log_path) {
  window_size_ = config.window_size;
  log_file_.open(log_path, std::ios::out);
  if (!log_file_.is_open()) {
    BAND_LOG_PROD(BAND_LOG_ERROR,
                  "ThermalEstimator failed to open the log file %s: %s",
                  log_path.c_str(), strerror(errno));
  } else {
    log_file_ << "[";
  }
  return absl::OkStatus();
}

ThermalEstimator::~ThermalEstimator() {
  if (log_file_.is_open()) {
    log_file_.seekp(-1, std::ios_base::end);
    log_file_ << "]";
    log_file_.close();
  }
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalMap therm_start,
                              ThermalMap therm_end, FreqMap freq,
                              double latency) {
  if (log_file_.is_open()) {
    log_file_ << "{";
    log_file_ << "\"key\":"
              << "\"" << key.ToString() << "\",";
    log_file_ << "\"therm_start\":" << ThermalMapToJson(therm_start) << ",";
    log_file_ << "\"therm_end\":" << ThermalMapToJson(therm_end) << ",";
    log_file_ << "\"expected\":false";
    log_file_ << "},";
  }

  const size_t num_sensors = EnumLength<SensorFlag>();
  const size_t num_devices = EnumLength<DeviceFlag>();
  Eigen::VectorXd old_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(therm_start, num_sensors);
  PrintVector("old_therm_vec", old_therm_vec);
  Eigen::VectorXd new_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(therm_end, num_sensors);
  PrintVector("new_therm_vec", new_therm_vec);
  Eigen::VectorXd freq_vec =
      ConvertTMapToEigenVector<FreqMap>(freq, num_devices);
  PrintVector("freq_vec", freq_vec);
  Eigen::VectorXd lat_vec = GetOneHotVector(
      latency, num_devices,
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));
  PrintVector("lat_vec", lat_vec);
  Eigen::VectorXd therm_lat_vec = old_therm_vec * latency;
  PrintVector("therm_lat_vec", therm_lat_vec);
  Eigen::VectorXd freq_3_vec =
      freq_vec.cwiseProduct(freq_vec.cwiseProduct(freq_vec));
  PrintVector("freq_3_vec", freq_3_vec);
  Eigen::VectorXd freq_3_lat_vec = freq_3_vec * latency;
  PrintVector("freq_3_lat_vec", freq_3_lat_vec);
  Eigen::VectorXd lat_fill_vec = GetFillVector(latency, num_devices);
  PrintVector("lat_fill_vec", lat_fill_vec);

  // num_sensors + num_devices + num_devices + num_devices
  size_t feature_size = old_therm_vec.size() + therm_lat_vec.size() +
                        freq_3_lat_vec.size() + lat_fill_vec.size();
  size_t target_size = new_therm_vec.size();

  Eigen::VectorXd feature(feature_size);
  feature << old_therm_vec, therm_lat_vec, freq_3_lat_vec, lat_fill_vec;

  features_.push_back({feature, new_therm_vec});
  if (features_.size() > window_size_) {
    features_.pop_front();
  }
  if (features_.size() < window_size_) {
    BAND_LOG_PROD(BAND_LOG_INFO,
                  "ThermalEstimator, Not enough data collected. Current number "
                  "of data: %d",
                  features_.size());
    return;
  }

  Eigen::MatrixXd data(window_size_, feature_size);
  Eigen::MatrixXd target(window_size_, target_size);
  for (int i = 0; i < window_size_; i++) {
    for (int j = 0; j < feature_size; j++) {
      data(i, j) = features_[i].first(j);
    }
    for (int j = 0; j < target_size; j++) {
      target(i, j) = features_[i].second(j);
    }
  }

  model_ = SolveLinear(data, target);
}

void ThermalEstimator::UpdateWithEvent(const SubgraphKey& key,
                                       size_t event_handle) {
  auto therm_interval = thermal_profiler_->GetInterval(event_handle);
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency =
      latency_profiler_->GetDuration<std::chrono::milliseconds>(event_handle);
  BAND_LOG_PROD(BAND_LOG_INFO, "Duration: %f", latency);
  Update(key, therm_interval.first.second, therm_interval.second.second,
         freq_interval.second.second, latency);
}

ThermalMap ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return ConvertEigenVectorToTMap<ThermalMap>(
      features_[features_.size() - 1].second);
}

ThermalMap ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  const size_t num_sensors = EnumLength<SensorFlag>();
  const size_t num_devices = EnumLength<DeviceFlag>();

  auto cur_therm_map = thermal_profiler_->GetAllThermal();
  auto cur_therm =
      ConvertTMapToEigenVector<ThermalMap>(cur_therm_map, num_sensors);
  auto cur_freq = ConvertTMapToEigenVector<FreqMap>(
      frequency_profiler_->GetAllFrequency(), EnumLength<DeviceFlag>());
  auto expected_latency = GetOneHotVector(
      frequency_latency_estimator_->GetExpected(key), num_devices,
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));

  size_t feature_size = cur_therm.size() + cur_freq.size() +
                        expected_latency.size() + expected_latency.size();
  Eigen::VectorXd feature(feature_size);
  feature << cur_therm, cur_freq, (cur_freq.cwiseProduct(expected_latency)),
      expected_latency;

  if (features_.size() < window_size_) {
    BAND_LOG_PROD(BAND_LOG_INFO,
                  "ThermalEstimator, Not enough data collected. Current number "
                  "of data: %d",
                  features_.size());
    return {};
  }

  auto expected_therm =
      ConvertEigenVectorToTMap<ThermalMap>(model_.transpose() * feature);
  if (log_file_.is_open()) {
    log_file_ << "{";
    log_file_ << "\"key\":\"" << key.ToString() << "\",";
    log_file_ << "\"therm_start\":" << ThermalMapToJson(cur_therm_map) << ",";
    log_file_ << "\"therm_end\":" << ThermalMapToJson(expected_therm) << ",";
    log_file_ << "\"expected\":true";
    log_file_ << "},";
  }
  return expected_therm;
}

absl::Status ThermalEstimator::LoadModel(std::string profile_path) {
  Json::Value root;
  std::ifstream file(profile_path);
  file >> root;
  window_size_ = root["window_size"].asInt();
  model_ = JsonToEigenMatrix(root["model"]);
  return absl::OkStatus();
}

absl::Status ThermalEstimator::DumpModel(std::string profile_path) {
  Json::Value root;
  root["window_size"] = window_size_;
  root["model"] = EigenMatrixToJson(model_);
  std::ofstream file(profile_path);
  file << root;
  return absl::OkStatus();
}

Json::Value ThermalEstimator::EigenMatrixToJson(Eigen::MatrixXd matrix) {
  Json::Value result;
  for (int i = 0; i < matrix.rows(); i++) {
    Json::Value row;
    for (int j = 0; j < matrix.cols(); j++) {
      row.append(matrix(i, j));
    }
    result.append(row);
  }
  return result;
}

Eigen::MatrixXd ThermalEstimator::JsonToEigenMatrix(Json::Value json) {
  Eigen::MatrixXd result(json.size(), json[0].size());
  for (int i = 0; i < json.size(); i++) {
    for (int j = 0; j < json[i].size(); j++) {
      result(i, j) = json[i][j].asDouble();
    }
  }
  return result;
}

}  // namespace band