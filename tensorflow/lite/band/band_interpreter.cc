
#include "tensorflow/lite/band/band_interpreter.h"

#include "tensorflow/lite/tools/logging.h"

namespace tflite {

namespace impl {

namespace {

BandInterpreter::BandInterpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter : DefaultErrorReporter()),
      lazy_delegate_provider_(
          TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {})) {

  TFLITE_LOG(INFO) << "Initialized TensorFlow Lite runtime.";

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
  own_external_cpu_backend_context_->
      set_internal_backend_context(
          std::make_unique<CpuBackendContext>());
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

}