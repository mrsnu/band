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

Eigen::VectorXd GetOneHotVector(double value, size_t size, size_t index) {
  Eigen::VectorXd vec(size);
  vec.setZero();
  vec(index) = value;
  return vec;
}

}  // anonymous namespace

ThermalEstimator::~ThermalEstimator() {
  {
    std::lock_guard<std::mutex> lock(model_update_queue_mutex_);
    model_update_thread_exit_ = true;
  }
  model_update_cv_.notify_one();
  model_update_thread_.join();
}

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  window_size_ = config.window_size;
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, Job& job) {
  BAND_TRACER_SCOPED_THREAD_EVENT(Update);
  auto start_time = job.start_time;
  auto end_time = job.end_time;
  auto& start_therm = job.start_thermal;
  auto& end_therm = job.end_thermal;

  auto start_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(job.start_thermal, num_sensors_);
  auto end_therm_vec =
      ConvertTMapToEigenVector<ThermalMap>(job.end_thermal, num_sensors_);

  FreqMap freq;
  freq[FreqFlag::kCPU] = 1;
  freq[FreqFlag::kGPU] = 1;
  freq[FreqFlag::kDSP] = 1;
  freq[FreqFlag::kNPU] = 1;
  freq[FreqFlag::kRuntime] = 1;
  auto freq_vec = ConvertTMapToEigenVector<FreqMap>(freq, num_devices_);

  model_update_queue_.push({key.GetWorkerId(), start_time, end_time,
                            start_therm_vec, end_therm_vec, freq_vec});
  model_update_cv_.notify_one();
}

void ThermalEstimator::UpdateModel() {
  BAND_TRACER_SCOPED_THREAD_EVENT(UpdateModel);
  if (window_size_ > features_.size()) {
    return;
  }

  auto data_size = window_size_ * (window_size_ - 1) / 2;

  Eigen::MatrixXd x(data_size, feature_size_);
  Eigen::MatrixXd y(data_size, num_sensors_);

  size_t index = 0;
  for (int i = 0; i < window_size_; i++) {
    auto& s_worker_id = std::get<0>(features_[i]);
    auto& s_start_time = std::get<1>(features_[i]);
    auto& s_end_time = std::get<2>(features_[i]);
    auto& s_start_therm_vec = std::get<3>(features_[i]);
    auto& s_freq_vec = std::get<5>(features_[i]);

    auto freq_3_vec =
        s_freq_vec.cwiseProduct(s_freq_vec.cwiseProduct(s_freq_vec));

    auto s_latency = (s_end_time - s_start_time) / 1000.f;

    auto freq_3_lat_vec = freq_3_vec * s_latency;
    auto freq_lat_vec = s_freq_vec * s_latency;
    auto lat_vec = GetOneHotVector(s_latency, num_devices_, s_worker_id);

    for (int j = i + 1; j < window_size_; j++) {
      Eigen::VectorXd feature(feature_size_);
      auto& e_worker_id = std::get<0>(features_[j]);
      auto& e_start_time = std::get<1>(features_[j]);
      auto& e_end_time = std::get<2>(features_[j]);
      auto& e_end_therm = std::get<4>(features_[j]);
      auto& e_freq = std::get<5>(features_[j]);

      auto e_freq_3_vec = e_freq.cwiseProduct(e_freq.cwiseProduct(e_freq));

      auto e_latency = (e_end_time - e_start_time) / 1000.f;
      auto e_freq_3_lat_vec = e_freq_3_vec * e_latency;
      auto e_freq_lat_vec = e_freq * e_latency;
      auto e_lat_vec = GetOneHotVector(e_latency, num_devices_, e_worker_id);

      auto total_latency = (e_end_time - s_start_time) / 1000.f;

      auto therm_lat_vec = s_start_therm_vec * total_latency;

      if (index == 0) {
        feature << therm_lat_vec, freq_3_lat_vec + e_freq_3_lat_vec,
            freq_lat_vec + e_freq_lat_vec, lat_vec + e_lat_vec;
      } else {
        // Accumulate
        feature << therm_lat_vec,
            x.row(index - 1).segment(num_sensors_, num_devices_).transpose() +
                e_freq_3_lat_vec,
            x.row(index - 1)
                    .segment(2 * num_devices_, num_devices_)
                    .transpose() +
                e_freq_lat_vec,
            x.row(index - 1)
                    .segment(3 * num_devices_, num_devices_)
                    .transpose() +
                e_lat_vec;
      }

      x.row(index) = feature;
      y.row(index) = (e_end_therm - s_start_therm_vec);
      index++;
    }
  }
  {
    BAND_TRACER_SCOPED_THREAD_EVENT(UpdateModelSolve);
    std::unique_lock<std::mutex> lock(model_mutex_);
    model_ = (x.transpose() * x).ldlt().solve(x.transpose() * y);
  }
}

void ThermalEstimator::ModelUpdateThreadLoop() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(model_update_queue_mutex_);
      model_update_cv_.wait(lock, [&] {
        return (!model_update_queue_.empty()) || model_update_thread_exit_;
      });

      if (model_update_thread_exit_) {
        break;
      }

      while (!model_update_queue_.empty()) {
        features_.push_back(model_update_queue_.front());
        model_update_queue_.pop();
      }
    }
    {
      if (features_.size() > window_size_) {
        features_.pop_front();
      }

      UpdateModel();
    }
  }
}

ThermalMap ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return profile_database_.at(key);
}

ThermalMap ThermalEstimator::GetExpected(const ThermalKey& thermal_key) const {
  // auto expected_therm =
  //     ConvertEigenVectorToTMap<ThermalMap>(model_.transpose() * feature);
  // return expected_therm;
  return {};
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
  UpdateModel();
  Json::Value root;
  root["window_size"] = window_size_;
  root["model"] = EigenMatrixToJson(model_);
  std::ofstream file(profile_path);
  file << root;
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

Eigen::VectorXd ThermalEstimator::GetFeatureVector(
    const Eigen::VectorXd& therm_vec, const Eigen::VectorXd& freq_vec,
    const Eigen::VectorXd& lat_vec, size_t worker_id,
    double total_latency) const {
  Eigen::VectorXd feature(feature_size_);

  Eigen::VectorXd therm_lat_vec = therm_vec * total_latency;
  Eigen::VectorXd freq_3_vec =
      freq_vec.cwiseProduct(freq_vec.cwiseProduct(freq_vec));
  Eigen::VectorXd freq_3_lat_vec = freq_3_vec.cwiseProduct(lat_vec);
  Eigen::VectorXd freq_lat_vec = freq_vec.cwiseProduct(lat_vec);

  feature << therm_lat_vec, freq_3_lat_vec, freq_lat_vec, lat_vec;
  return feature;
}

}  // namespace band