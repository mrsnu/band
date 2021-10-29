
#include "tensorflow/lite/band/band_interpreter.h"

#include "tensorflow/lite/minimal_logging.h"

namespace tflite {

namespace impl {

BandInterpreter::BandInterpreter(ErrorReporter* error_reporter) {
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");

  InitBackendContext();
}

void BandInterpreter::InitBackendContext() {
  // Reserve some space for the tensors to avoid excessive resizing.
  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  // This operation is cheap because we allocate the CPU context resources (i.e.
  // threads) lazily.
  own_external_cpu_backend_context_.reset(new ExternalCpuBackendContext());
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();

  // Initialize internal backend context for cpu contexts
//  own_external_cpu_backend_context_->
//      set_internal_backend_context(
//          std::make_unique<CpuBackendContext>());
}

BandInterpreter::~BandInterpreter() {
  Interpreter::~Interpreter();

  // update the profile file to include all new profile results from this run
  // profiling::util::UpdateDatabase(profile_database_, model_configs_,
  //                                 profile_database_json_);
  // WriteJsonObjectToFile(profile_database_json_, profile_data_path_);
}


}

}

