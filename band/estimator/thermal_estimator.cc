#include "band/estimator/thermal_estimator.h"

#include <fstream>

#include "absl/strings/str_format.h"
#include "band/common.h"
#include "band/engine_interface.h"
#include "band/job_tracer.h"
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

#ifdef BAND_SPLASH
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

Eigen::VectorXd GetFeatureVector(const ThermalMap& therm, const FreqMap freq,
                                 size_t worker_id, double latency) {
  Eigen::VectorXd feature(feature_size_);
  Eigen::VectorXd therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(therm, num_sensors_);
  Eigen::VectorXd freq_vec =
      ConvertTMapToEigenVector<FreqMap>(freq, num_devices_);
  Eigen::VectorXd latency_vec =
      GetOneHotVector(latency, num_devices_, worker_id);
  Eigen::VectorXd therm_lat_vec = therm_vec * latency;
  Eigen::VectorXd freq_3_vec =
      freq_vec.cwiseProduct(freq_vec.cwiseProduct(freq_vec));
  Eigen::VectorXd freq_3_lat_vec = freq_3_vec * latency;
  Eigen::VectorXd lat_fill_vec = GetFillVector(latency, num_devices_);

  feature << therm_vec, therm_lat_vec, freq_3_lat_vec, lat_fill_vec;
  return feature;
}

#endif  // BAND_SPLASH

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  JobTracer::Get().AddStream("ThermalEstimator");
#ifdef BAND_SPLASH
  window_size_ = config.window_size;
#endif  // BAND_SPLASH
  return absl::OkStatus();
}

void ThermalEstimator::Update(const ThermalKey& key, ThermalMap target_therm) {
  auto trace_handle = JobTracer::Get().BeginEvent("ThermalEstimator", "Update");
  profile_database_[key] = therm_end;

#ifdef BAND_SPLASH
  auto therm_start = std::get<0>(key);
  auto freq_start = std::get<1>(key);
  auto subgraph_key = std::get<2>(key);

  Eigen::VectorXd new_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(therm_end, num_sensors);
  Eigen::VectorXd feature =
      GetFeatureVector(therm_start, freq_start, subgraph_key.GetWorkerId(),
                       latency_estimator_->GetExpected(subgraph_key) / 1000.f);

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
  JobTracer::Get().EndEvent("ThermalEstimator", trace_handle);
}

Eigen::MatrixXd ThermalEstimator::SolveLinear(Eigen::MatrixXd& x,
                                              Eigen::MatrixXd& y) {
  return (x.transpose() * x).ldlt().solve(x.transpose() * y);
}

void ThermalEstimator::UpdateWithEvent(const SubgraphKey& key,
                                       size_t event_handle) {
  auto therm_interval = thermal_profiler_->GetInterval(event_handle);
  auto freq_interval = frequency_profiler_->GetInterval(event_handle);
  auto latency =
      latency_profiler_->GetDuration<std::chrono::microseconds>(event_handle) /
      1000.f;
  Update(key, therm_interval.first.second, therm_interval.second.second,
         freq_interval.second.second, latency);
}

ThermalMap ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return profile_database_.at(key);
}

ThermalMap ThermalEstimator::GetExpected(const ThermalKey& thermal_key) const {
  auto trace_handle =
      JobTracer::Get().BeginEvent("ThermalEstimator", "GetExpected");
#ifdef BAND_SPLASH
  auto cur_therm_map = std::get<0>(thermal_key);
  auto cur_freq_map = std::get<1>(thermal_key);
  auto key = std::get<2>(thermal_key);

  profile_database_[key] = cur_therm_map;

  Eigen::VectorXd feature =
      GetFeatureVector(cur_therm_map, cur_freq_map, key.GetWorkerId(),
                       latency_estimator_->GetExpected(key) / 1000.f);

  auto expected_therm =
      ConvertEigenVectorToTMap<ThermalMap>(model_.transpose() * feature);
  JobTracer::Get().EndEvent("ThermalEstimator", trace_handle);
  return expected_therm;
#else
  JobTracer::Get().EndEvent("ThermalEstimator", trace_handle);
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

Json::Value ThermalEstimator::EigenMatrixToJson(Eigen::MatrixXd& matrix) {
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