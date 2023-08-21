#include "band/estimator/thermal_estimator.h"

#include "absl/strings/str_format.h"
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

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  window_size_ = config.window_size;
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalMap therm_start,
                              ThermalMap therm_end, FreqMap freq,
                              double latency) {
  const size_t num_sensors = EnumLength<SensorFlag>();
  const size_t num_devices = EnumLength<DeviceFlag>();
  Eigen::VectorXd old_therm =
      ConvertTMapToEigenVector<ThermalMap>(therm_start, num_sensors);
  Eigen::VectorXd new_therm =
      ConvertTMapToEigenVector<ThermalMap>(therm_end, num_sensors);
  Eigen::VectorXd freq_info =
      ConvertTMapToEigenVector<FreqMap>(freq, num_devices);
  Eigen::VectorXd latency_vector = GetOneHotVector(
      latency, num_devices,
      static_cast<size_t>(engine_->GetWorkerDevice(key.GetWorkerId())));
  BAND_LOG_PROD(BAND_LOG_INFO, "Old therm shape: %d", old_therm.size());
  BAND_LOG_PROD(BAND_LOG_INFO, "New therm shape: %d", new_therm.size());
  BAND_LOG_PROD(BAND_LOG_INFO, "Freq shape: %d", freq_info.size());
  BAND_LOG_PROD(BAND_LOG_INFO, "Latency shape: %d", latency_vector.size());
  BAND_LOG_PROD(BAND_LOG_INFO, "Worker: %d, Device: %s", key.GetWorkerId(),
                ToString(engine_->GetWorkerDevice(key.GetWorkerId())));

  // num_sensors + num_devices + num_devices
  size_t feature_size = old_therm.size() + freq_info.size() +
                        latency_vector.size() + latency_vector.size();
  size_t target_size = new_therm.size();

  Eigen::VectorXd feature(feature_size);
  feature << old_therm, freq_info, (freq_info.cwiseProduct(latency_vector)),
      latency_vector;
  BAND_LOG_PROD(BAND_LOG_INFO, "Feature shape: %d", feature.size());

  features_.push_back({feature, new_therm});
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
  BAND_LOG_PROD(BAND_LOG_INFO, "Data shape: %d, %d", data.rows(), data.cols());
  BAND_LOG_PROD(BAND_LOG_INFO, "Target shape: %d, %d", target.rows(),
                target.cols());
  BAND_LOG_PROD(BAND_LOG_INFO, "Feature shape: %d, %d", feature.rows(),
                feature.cols());
  for (int i = 0; i < window_size_; i++) {
    for (int j = 0; j < feature_size; j++) {
      data(i, j) = features_[i].first(j);
    }
    for (int j = 0; j < target_size; j++) {
      target(i, j) = features_[i].second(j);
    }
  }

  BAND_LOG_PROD(BAND_LOG_INFO, "Model shape: %d, %d", model_.rows(),
                model_.cols());
  model_ = SolveLinear(data, target);
  // Print data
  BAND_LOG_PROD(BAND_LOG_INFO, "Data: ");
  for (int i = 0; i < data.rows(); i++) {
    std::string line = "";
    for (int j = 0; j < data.cols(); j++) {
      line += absl::StrFormat("%f, ", data(i, j));
    }
    BAND_LOG_PROD(BAND_LOG_INFO, "%s", line.c_str());
  }
  // Print target
  BAND_LOG_PROD(BAND_LOG_INFO, "Target: ");
  for (int i = 0; i < target.rows(); i++) {
    std::string line = "";
    for (int j = 0; j < target.cols(); j++) {
      line += absl::StrFormat("%f, ", target(i, j));
    }
    BAND_LOG_PROD(BAND_LOG_INFO, "%s", line.c_str());
  }
  // Print model
  BAND_LOG_PROD(BAND_LOG_INFO, "Model: ");
  for (int i = 0; i < model_.rows(); i++) {
    std::string line = "";
    for (int j = 0; j < model_.cols(); j++) {
      line += absl::StrFormat("%f, ", model_(i, j));
    }
    BAND_LOG_PROD(BAND_LOG_INFO, "%s", line.c_str());
  }
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

  auto cur_therm = ConvertTMapToEigenVector<ThermalMap>(
      thermal_profiler_->GetAllThermal(), num_sensors);
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
  BAND_LOG_PROD(BAND_LOG_INFO, "Model shape: %d, %d", model_.rows(),
                model_.cols());
  BAND_LOG_PROD(BAND_LOG_INFO, "Current thermal shape: %d", feature.size());

  auto expected_therm =
      ConvertEigenVectorToTMap<ThermalMap>(model_.transpose() * feature);
  BAND_LOG_PROD(BAND_LOG_INFO, "Expected thermal shape: %d",
                expected_therm.size());
  BAND_LOG_PROD(BAND_LOG_INFO, "Expected thermal: ");
  for (const auto& pair : expected_therm) {
    BAND_LOG_PROD(BAND_LOG_INFO, "%s: %f", ToString(pair.first), pair.second);
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