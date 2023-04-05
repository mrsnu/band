#include "band/backend/tfl/backend.h"

namespace band {
bool TfLiteRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      BackendType::TfLite, new tfl::ModelExecutorCreator, new tfl::ModelCreator,
      new tfl::UtilCreator);
  return true;
}
}  // namespace band
