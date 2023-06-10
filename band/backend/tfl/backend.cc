#include "band/backend/tfl/backend.h"

namespace band {
bool TfLiteRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      BackendType::kBandTfLite, new tfl::ModelExecutorCreator, new tfl::ModelCreator,
      new tfl::UtilCreator);
  return true;
}
}  // namespace band
