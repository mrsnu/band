#ifndef BAND_BACKEND_FACTORY_H_
#define BAND_BACKEND_FACTORY_H_

#include "band/backend_factory.h"
#include "band/common.h"
#include "band/interface/interpreter.h"
#include "band/interface/model.h"

// Expected workflow (per each backend)
// 1. Implement creators
// 2. Register them with `RegisterBackendCreators` in arbitrary global function
// (e.g., TfLite::RegisterCreators)
// 3. Add an arbitrary global function to `RegisterBackendInternal` with
// corresponding flag

namespace Band {

template <typename Base, class... Args>
struct Creator {
 public:
  virtual Base* Create(Args...) const { return nullptr; };
};

class BackendFactory {
 public:
  static Interface::IInterpreter* CreateInterpreter(BandBackendType backend);
  static Interface::IModel* CreateModel(BandBackendType backend, ModelId id);
  static Interface::IBackendUtil* GetBackendUtil(BandBackendType backend);
  static std::vector<BandBackendType> GetAvailableBackends();

  static void RegisterBackendCreators(
      BandBackendType backend,
      Creator<Interface::IInterpreter>* interpreter_creator,
      Creator<Interface::IModel, ModelId>* model_creator,
      Creator<Interface::IBackendUtil>* util_creator);

 private:
  BackendFactory() = default;

  static std::map<BandBackendType,
                  std::shared_ptr<Creator<Interface::IInterpreter>>>
      interpreter_creators_;
  static std::map<BandBackendType,
                  std::shared_ptr<Creator<Interface::IModel, ModelId>>>
      model_creators_;
  static std::map<BandBackendType,
                  std::shared_ptr<Creator<Interface::IBackendUtil>>>
      util_creators_;
};
}  // namespace Band

#endif