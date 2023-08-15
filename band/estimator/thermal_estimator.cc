#include "band/estimator/thermal_estimator.h"

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler.h"
#include "band/worker.h"

namespace band {

namespace {

Eigen::VectorXd ConvertThermalValueToEigenVector(
    const band::ThermalValue& value) {
  Eigen::VectorXd vec(value.size());
  int i = 0;
  for (const auto& pair : value) {
    vec(i) = pair.second;
    i++;
  }
  return vec;
}

ThermalValue ConvertEigenVectorToThermalValue(const Eigen::VectorXd& vec) {
  ThermalValue value;
  for (int i = 0; i < vec.size(); i++) {
    value[static_cast<DeviceFlag>(i)] = vec(i);
  }
  return value;
}

}  // anonymous namespace

absl::Status ThermalEstimator::Init(const ThermalProfileConfig& config) {
  // Register thermal resources
  for (int i = 0;; i++) {
    auto status =
        resource_monitor_->AddThermalResource(ThermalFlag::TZ_TEMPERATURE, 0);
    if (!status.ok()) {
      num_resources_ = i;
      break;
    }
  }
  return absl::OkStatus();
}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalValue thermal) {}

void ThermalEstimator::Update(const SubgraphKey& key, ThermalValue old_value,
                              ThermalValue new_value) {
  Eigen::VectorXd old_vec = ConvertThermalValueToEigenVector(old_value);
  Eigen::VectorXd new_vec = ConvertThermalValueToEigenVector(new_value);
  ConvertEigenVectorToThermalValue(new_vec);
}

absl::Status ThermalEstimator::Load(ModelId model_id,
                                    std::string profile_path) {
  return absl::OkStatus();
}

absl::Status ThermalEstimator::Profile(ModelId model_id) {
  auto profile_thermal_func = [&](const SubgraphKey& subgraph_key) -> void {

  };

  for (WorkerId worker_id = 0; worker_id < engine_->GetNumWorkers();
       worker_id++) {
    Worker* worker = engine_->GetWorker(worker_id);
    worker->Pause();
    worker->Wait();
    std::thread profile_thread([&]() {
#if BAND_IS_MOBILE
      if (worker->GetWorkerThreadAffinity().NumEnabled() > 0 &&
          !SetCPUThreadAffinity(worker->GetWorkerThreadAffinity()).ok()) {
        return absl::InternalError(
            absl::StrFormat("Failed to propagate thread affinity of worker id "
                            "%d to profile thread",
                            worker_id));
      }
#endif  // BAND_IS_MOBILE

      engine_->ForEachSubgraph(profile_thermal_func);
    });
  }
  for (int i = 0; i < num_resources_; i++) {
    auto status_or_thermal =
        resource_monitor_->GetThermal(ThermalFlag::TZ_TEMPERATURE, i);
    if (!status_or_thermal.ok()) {
      return status_or_thermal.status();
    }
    auto thermal_value = status_or_thermal.value();
  }
  return absl::OkStatus();
}

ThermalValue ThermalEstimator::GetProfiled(const SubgraphKey& key) const {
  return {};
}

ThermalValue ThermalEstimator::GetExpected(const SubgraphKey& key) const {
  return {};
}

absl::Status ThermalEstimator::DumpProfile() { return absl::OkStatus(); }

}  // namespace band