#ifndef TENSORFLOW_LITE_PROFILING_FUNCTION_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_FUNCTION_PROFILER_H_

#include <string>

namespace tflite {
namespace profiling {

// Profiler class for average latency computation
class FunctionProfiler {
 public:
  FunctionProfiler(std::string function_name);
  ~FunctionProfiler();

 private:
  int64_t function_start_time_;
  std::string function_name_;
};

#define TFLITE_MEASURE_FUNCTION_DURATION()          \
  tflite::profiling::FunctionProfiler(__PRETTY_FUNCTION__)
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_FUNCTION_PROFILER_H_
