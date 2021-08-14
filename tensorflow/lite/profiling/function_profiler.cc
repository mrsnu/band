#include "tensorflow/lite/profiling/function_profiler.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace profiling {

FunctionProfiler::FunctionProfiler(std::string function_name)
  : function_name_(function_name) {
  function_start_time_ = time::NowNanos();
}

FunctionProfiler::~FunctionProfiler() {
  int64_t current_time = time::NowNanos();
  TFLITE_LOG(INFO) << function_name_ << " took " << current_time - function_start_time_ << " (ns)";
}


} // namespace profiling
} // namespace tflite 
