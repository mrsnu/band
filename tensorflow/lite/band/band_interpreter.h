#ifndef TENSORFLOW_LITE_BAND_INTERPRETER_H_
#define TENSORFLOW_LITE_BAND_INTERPRETER_H_

#include "tensorflow/lite/interpreter.h"

namespace tflite {

namespace delegates {
}
namespace impl {

class BandInterpreter : public Interpreter {
  public:

    explicit BandInterpreter(ErrorReporter* error_reporter, RuntimeConfig runtime_config) override;
    
    ~BandInterpreter() override;

  private:

  void InitBackendContext()

}


}
#endif  // TENSORFLOW_LITE_BAND_INTERPRETER_H_