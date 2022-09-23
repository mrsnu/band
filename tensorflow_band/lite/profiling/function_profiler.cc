#include "tensorflow/lite/profiling/function_profiler.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace profiling {

FunctionProfiler::FunctionProfiler(std::string function_name)
  : function_name_(function_name) {
  function_start_time_ = time::NowMicros();
}

FunctionProfiler::~FunctionProfiler() {
  int64_t current_time = time::NowMicros();
  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "%s took %d (us)", function_name_.c_str(), current_time - function_start_time_);
}


} // namespace profiling
} // namespace tflite 
