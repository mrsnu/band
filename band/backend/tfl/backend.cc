#include "band/backend/tfl/backend.h"

namespace Band {
bool TfLiteRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      BackendType::TfLite, new TfLite::ModelExecutorCreator, new TfLite::ModelCreator,
      new TfLite::UtilCreator);
  return true;
}
}  // namespace Band
