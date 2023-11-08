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

// std::string ThermalMapToJson(ThermalMap thermal_map) {
//   std::string result = "{";
//   for (auto& pair : thermal_map) {
//     result += "\"" + std::string(ToString(pair.first)) + "\":";
//     result += std::to_string(pair.second);
//     result += ",";
//   }
//   result.pop_back();
//   result += "}";
//   return result;
// }

// void PrintVector(std::string name, const Eigen::VectorXd& vec) {
//   BAND_LOG_PROD(BAND_LOG_INFO, "%s: ", name.c_str());
//   for (int i = 0; i < vec.size(); i++) {
//     BAND_LOG_PROD(BAND_LOG_INFO, "%f ", vec(i));
//   }
// }

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
#ifdef BAND_SPLASH
  window_size_ = config.window_size;
#endif  // BAND_SPLASH
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalMap therm_start,
                              ThermalMap therm_end, FreqMap freq,
                              double latency) {
  profile_database_[key] = therm_end;

#ifdef BAND_SPLASH
  const size_t num_sensors = EnumLength<SensorFlag>();
  const size_t num_devices = EnumLength<DeviceFlag>();
  Eigen::VectorXd old_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(therm_start, num_sensors);
  Eigen::VectorXd new_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(therm_end, num_sensors);
  Eigen::VectorXd freq_vec =
      ConvertTMapToEigenVector<FreqMap>(freq, num_devices);
  Eigen::VectorXd lat_vec = GetOneHotVector(
      latency, num_devices,
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));
  Eigen::VectorXd therm_lat_vec = old_therm_vec * latency;
  Eigen::VectorXd freq_3_vec =
      freq_vec.cwiseProduct(freq_vec.cwiseProduct(freq_vec));
  Eigen::VectorXd freq_3_lat_vec = freq_3_vec * latency;
  Eigen::VectorXd lat_fill_vec = GetFillVector(latency, num_devices);

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

  size_t window_size = std::min(window_size_, features_.size());
  Eigen::MatrixXd data(window_size, feature_size);
  Eigen::MatrixXd target(window_size, target_size);
  for (int i = 0; i < window_size; i++) {
    for (int j = 0; j < feature_size; j++) {
      data(i, j) = features_[i].first(j);
    }
    for (int j = 0; j < target_size; j++) {
      target(i, j) = features_[i].second(j);
    }
  }

  model_ = SolveLinear(data, target);
#endif  // BAND_SPLASH
}

void ThermalEstimator::UpdateWithEvent(const SubgraphKey& key,
                                       size_t event_handle) {
  auto therm_interval = thermal_profiler_->GetInterval(event_handle);
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency =
      latency_profiler_->GetDuration<std::chrono::milliseconds>(event_handle);
  Update(key, therm_interval.first.second, therm_interval.second.second,
         freq_interval.second.second, latency);
}

ThermalMap ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return profile_database_.at(key);
}

ThermalMap ThermalEstimator::GetExpected(const SubgraphKey& key) const {
#ifdef BAND_SPLASH
  const size_t num_sensors = EnumLength<SensorFlag>();
  const size_t num_devices = EnumLength<DeviceFlag>();
  double latency = latency_estimator_->GetExpected(key);
  auto cur_freq_map = frequency_profiler_->GetAllFrequency();
  auto cur_therm_map = thermal_profiler_->GetAllThermal();
  profile_database_[key] = cur_therm_map;

  Eigen::VectorXd old_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(cur_therm_map, num_sensors);
  Eigen::VectorXd freq_vec =
      ConvertTMapToEigenVector<FreqMap>(cur_freq_map, num_devices);
  Eigen::VectorXd latency_vec = GetOneHotVector(
      latency, num_devices,
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));
  Eigen::VectorXd therm_lat_vec = old_therm_vec * latency;
  Eigen::VectorXd freq_3_vec =
      freq_vec.cwiseProduct(freq_vec.cwiseProduct(freq_vec));
  Eigen::VectorXd freq_3_lat_vec = freq_3_vec * latency;
  Eigen::VectorXd lat_fill_vec = GetFillVector(latency, num_devices);

  size_t feature_size = old_therm_vec.size() + therm_lat_vec.size() +
                        freq_3_lat_vec.size() + lat_fill_vec.size();
  Eigen::VectorXd feature(feature_size);
  feature << old_therm_vec, therm_lat_vec, freq_3_lat_vec, lat_fill_vec;

  auto expected_therm =
      ConvertEigenVectorToTMap<ThermalMap>(model_.transpose() * feature);
  return expected_therm;
#else
  return {};
#endif  // BAND_SPLASH
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
#ifdef BAND_SPLASH
  Json::Value root;
  root["window_size"] = window_size_;
  root["model"] = EigenMatrixToJson(model_);
  std::ofstream file(profile_path);
  file << root;
#endif  // BAND_SPLASH
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