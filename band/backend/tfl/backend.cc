#include "band/backend/tfl/backend.h"

namespace Band {
bool TfLiteRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      kBandTfLite, new TfLite::InterpreterCreator, new TfLite::ModelCreator,
      new TfLite::UtilCreator);
  return true;
}
}  // namespace Band
