#include "band/backend_factory.h"

#include <mutex>

#include "band/logger.h"

namespace Band {
using namespace Interface;

__attribute__((weak)) extern void TfLiteRegisterCreators() {}

// Expected process

void RegisterBackendInternal() {
  static std::once_flag g_flag;
  std::call_once(g_flag, [] {
    BAND_LOG_INTERNAL(BAND_LOG_INFO, "Register TFL backend");
    TfLiteRegisterCreators();
  });
}

std::map<BandBackendType, std::shared_ptr<Creator<IInterpreter>>>
    BackendFactory::interpreter_creators_ = {};
std::map<BandBackendType, std::shared_ptr<Creator<IModel, ModelId>>>
    BackendFactory::model_creators_ = {};
std::map<BandBackendType, std::shared_ptr<Creator<IBackendUtil>>>
    BackendFactory::util_creators_ = {};

absl::StatusOr<IInterpreter*> BackendFactory::CreateInterpreter(
    BandBackendType backend) {
  RegisterBackendInternal();
  auto it = interpreter_creators_.find(backend);
  if (it == interpreter_creators_.end()) {
    return absl::InternalError("Failed to create interpreter.");
  }
  return it->second->Create();
}

absl::StatusOr<IModel*> BackendFactory::CreateModel(BandBackendType backend,
                                                    ModelId id) {
  RegisterBackendInternal();
  auto it = model_creators_.find(backend);
  if (it == model_creators_.end()) {
    absl::InternalError("Failed to create model.");  
  }
  return it->second->Create(id);
}

absl::StatusOr<IBackendUtil*> BackendFactory::GetBackendUtil(
    BandBackendType backend) {
  RegisterBackendInternal();
  auto it = util_creators_.find(backend);
  if (it == util_creators_.end()) {
    return absl::InternalError("Failed to create backend util.");
  }
  return it->second->Create();
}

std::vector<BandBackendType> BackendFactory::GetAvailableBackends() {
  RegisterBackendInternal();
  // assume static creators are all valid - after instantiation to reach here
  std::vector<BandBackendType> valid_backends;

  for (auto type_interpretor_creator : interpreter_creators_) {
    valid_backends.push_back(type_interpretor_creator.first);
  }

  return valid_backends;
}

void BackendFactory::RegisterBackendCreators(
    BandBackendType backend, Creator<IInterpreter>* interpreter_creator,
    Creator<IModel, ModelId>* model_creator,
    Creator<IBackendUtil>* util_creator) {
  interpreter_creators_[backend] =
      std::shared_ptr<Creator<IInterpreter>>(interpreter_creator);
  model_creators_[backend] =
      std::shared_ptr<Creator<IModel, ModelId>>(model_creator);
  util_creators_[backend] =
      std::shared_ptr<Creator<IBackendUtil>>(util_creator);
}
}  // namespace Band