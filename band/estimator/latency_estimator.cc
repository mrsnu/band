#include "band/estimator/latency_estimator.h"

#include "absl/strings/str_format.h"
#include "band/engine_interface.h"
#include "band/json_util.h"
#include "band/logger.h"
#include "band/model_spec.h"
#include "band/profiler/latency_profiler.h"
#include "band/worker.h"

namespace band {

namespace {

std::set<int> UnitIndicesFromString(std::string unit_indices_string) {
  std::set<int> unit_indices;
  std::stringstream ss(unit_indices_string);
  for (int i; ss >> i;) {
    unit_indices.insert(i);
    if (ss.peek() == ',') {
      ss.ignore();
    }
  }
  return unit_indices;
}

}  // anonymous namespace

absl::Status LatencyEstimator::Init(const LatencyProfileConfig& config) {
  profile_smoothing_factor_ = config.smoothing_factor;
  return absl::OkStatus();
}

void LatencyEstimator::Update(const SubgraphKey& key, double latency) {
  auto it = profile_database_.find(key);
  if (it == profile_database_.end()) {
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Initial profiled latency %s: %f.",
                      key.ToString().c_str(), latency);
    profile_database_[key] = {latency, latency};
    return;
  }
  double prev_latency = it->second.moving_averaged;
  profile_database_[key].moving_averaged =
      profile_smoothing_factor_ * latency +
      (1 - profile_smoothing_factor_) * prev_latency;
}

void LatencyEstimator::UpdateWithEvent(const SubgraphKey& key,
                                       size_t event_handle) {
  Update(key, latency_profiler_->GetDuration<std::chrono::milliseconds>(
                  event_handle));
}

double LatencyEstimator::GetProfiled(const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second.profiled;
  } else {
    BAND_LOG_PROD(BAND_LOG_INFO,
                  "[LatencyEstimator::GetProfiled] The given %s not found",
                  key.ToString().c_str());
    return 0;
  }
}

double LatencyEstimator::GetExpected(const SubgraphKey& key) const {
  auto it = profile_database_.find(key);
  if (it != profile_database_.end()) {
    return it->second.moving_averaged;
  } else {
    BAND_LOG_PROD(BAND_LOG_INFO,
                  "[LatencyEstimator::GetExpected] The given %s not found",
                  key.ToString().c_str());
    return 0;
  }
}

size_t LatencyEstimator::GetProfileHash() const {
  auto hash_func = std::hash<int>();
  size_t hash = hash_func(engine_->GetNumWorkers());
  for (int i = 0; i < engine_->GetNumWorkers(); i++) {
    hash ^= hash_func(static_cast<int>(engine_->GetWorker(i)->GetDeviceFlag()));
    hash ^= hash_func(engine_->GetWorker(i)->GetNumThreads());
    hash ^= hash_func(static_cast<int>(
        engine_->GetWorker(i)->GetWorkerThreadAffinity().GetCPUMaskFlag()));
  }
  return hash;
}

absl::Status LatencyEstimator::LoadModel(std::string profile_path) {
  if (!device::IsFileAvailable(profile_path)) {
    return absl::InternalError(absl::StrFormat(
        "Profile file %s does not exist.", profile_path.c_str()));
  }

  Json::Value profile_database_json = json::LoadFromFile(profile_path);
  if (profile_database_json["hash"].asUInt64() != GetProfileHash()) {
    return absl::InternalError(absl::StrFormat(
        "Profile hash mismatch. Expected %d, got %d.", GetProfileHash(),
        profile_database_json["hash"].asUInt64()));
  }

  Json::Value models_json = profile_database_json["models"];
  for (auto model_it = models_json.begin(); model_it != models_json.end();
       model_it++) {
    size_t model_id = model_it.key().asUInt();
    const Json::Value model_json = *model_it;
    for (auto unit_it = model_json.begin(); unit_it != model_json.end();
         unit_it++) {
      std::string unit_indices_string = unit_it.key().asString();
      std::set<int> unit_indices = UnitIndicesFromString(unit_indices_string);
      const Json::Value unit_json = *unit_it;
      for (auto worker_it = unit_json.begin(); worker_it != unit_json.end();
           worker_it++) {
        int worker_id = worker_it.key().asInt();
        const Json::Value worker_json = *worker_it;
        SubgraphKey key(model_id, worker_id, unit_indices);
        profile_database_[key] = {
            worker_json[worker_id]["profiled"].asDouble(),
            worker_json[worker_id]["moving_averaged"].asDouble()};
      }
    }
  }

  return absl::OkStatus();
}

absl::Status LatencyEstimator::DumpModel(std::string profile_path) {
  return json::WriteToFile(ProfileToJson(), profile_path);
}

Json::Value LatencyEstimator::ProfileToJson() {
  Json::Value profile_database_json;
  Json::Value models_json;
  profile_database_json["hash"] = GetProfileHash();
  for (auto& pair : profile_database_) {
    SubgraphKey key = pair.first;
    const int model_id = key.GetModelId();

    auto model_spec = engine_->GetModelSpec(model_id);
    if (model_spec == nullptr || model_spec->path.empty()) {
      BAND_LOG_INTERNAL(BAND_LOG_ERROR,
                        "Cannot find model %d from "
                        "model_configs. Will ignore.",
                        model_id);
      continue;
    }

    // copy all entries in id_profile --> database_json
    models_json[model_id][key.GetUnitIndicesString()][key.GetWorkerId()]
               ["profiled"] = pair.second.profiled;
    models_json[model_id][key.GetUnitIndicesString()][key.GetWorkerId()]
               ["moving_averaged"] = pair.second.moving_averaged;
  }
  models_json["models"] = models_json;
  return profile_database_json;
}

}  // namespace band