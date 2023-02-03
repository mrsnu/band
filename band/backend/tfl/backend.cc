#include "band/backend/tfl/backend.h"

namespace Band {
bool TfLiteRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      kBandTfLite, new TfLite::ModelExecutorCreator, new TfLite::ModelCreator,
      new TfLite::UtilCreator);
  return true;
}
}  // namespace Band
