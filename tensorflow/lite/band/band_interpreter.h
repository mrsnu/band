#ifndef TENSORFLOW_LITE_BAND_INTERPRETER_H_
#define TENSORFLOW_LITE_BAND_INTERPRETER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"  // IWYU pragma: export
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {

namespace impl {

class BandInterpreter : public Interpreter {
  public:

    explicit BandInterpreter(ErrorReporter* error_reporter);
    
    ~BandInterpreter();

  private:

  void InitBackendContext();

};


}
}
#endif  // TENSORFLOW_LITE_BAND_INTERPRETER_H_
