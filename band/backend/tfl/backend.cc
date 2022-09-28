#include "band/backend/tfl/backend.h"

namespace Band {
void TfLiteRegisterCreators() {
  BackendFactory::RegisterBackendCreators(
      kBandTfLite, new TfLite::InterpreterCreator, new TfLite::ModelCreator,
      new TfLite::UtilCreator);
}
}  // namespace Band